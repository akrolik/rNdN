#include "Backend/Scheduler/ListBlockScheduler.h"

#include "Backend/Scheduler/HardwareProperties.h"

#include "SASS/Analysis/Dependency/BlockDependencyAnalysis.h"

#include "Utils/Chrono.h"
#include "Utils/Options.h"

#include "Libraries/robin_hood.h"

namespace Backend {
namespace Scheduler {

void ListBlockScheduler::ScheduleBlock(SASS::BasicBlock *block)
{
	// Build the instruction dependency graphs (DAG) for the basic block, each
	// representing a schedulable section that may be reordered

	SASS::Analysis::BlockDependencyAnalysis dependencyAnalysis(m_function);
	dependencyAnalysis.Build(block);

	auto timeScheduler_start = Utils::Chrono::Start("List scheduler '" + block->GetName() + "'");

	// Get scheduler options
	
	auto optionReuse = Utils::Options::IsBackendSchedule_Reuse();
	auto optionDual = Utils::Options::IsBackendSchedule_Dual();
	auto optionCBarrier = Utils::Options::IsBackendSchedule_CBarrier();

	// Break ties using the instruction order, groups together related instructions

	auto& scheduledInstructions = block->GetInstructions();
	auto instructionCount = scheduledInstructions.size();

	robin_hood::unordered_map<SASS::Instruction *, std::uint32_t> instructionOrder;
	instructionOrder.reserve(instructionCount);

	auto orderValue = 0;
	for (auto& instruction : scheduledInstructions)
	{
		instructionOrder.emplace(instruction, orderValue++);
	}

	// Build the schedule for each schedulable section individually (guarantees ordering)

	scheduledInstructions.clear();
	scheduledInstructions.reserve(instructionCount);

	// For each barrier, keep track of the number of queued instructions, and the current wait position

	constexpr unsigned int BARRIER_COUNT = 6;
	const SASS::Schedule::Barrier BARRIER_MAP[BARRIER_COUNT] = {
		SASS::Schedule::Barrier::SB0,
		SASS::Schedule::Barrier::SB1,
		SASS::Schedule::Barrier::SB2,
		SASS::Schedule::Barrier::SB3,
		SASS::Schedule::Barrier::SB4,
		SASS::Schedule::Barrier::SB5
	};
#define BARRIER_UNMAP(barrier) static_cast<std::underlying_type_t<SASS::Schedule::Barrier>>(barrier)

	std::array<std::uint8_t, BARRIER_COUNT> barrierCount{};
	std::array<std::uint8_t, BARRIER_COUNT> barrierWait{};

	// Schedule the instructions based on the dependency DAG and hardware properties:
	//   - Priority function: lowest stall count
	//   - Pipeline depth & latencies

	// For each instruction maintain:
	//   - Earliest virtual clock cycle for execution (updated when each parent is scheduled)
	//   - Stall count required if scheduled next
	// Together these give the legal execution time for the instruction
	//
	// To account for variable latency dependencies, we use a barrier time to hint at the expected execution

	struct InstructionProperties
	{
	public:
		InstructionProperties(std::uint32_t time, std::uint32_t stall, std::uint32_t barrierTime, std::uint32_t barrierStall, std::uint32_t dependencyCount) :
			m_time(time), m_stall(stall), m_barrierTime(barrierTime), m_barrierStall(stall), m_dependencyCount(dependencyCount) {}

		std::uint32_t GetAvailableTime() const { return m_time; }
		void SetAvailableTime(std::uint32_t time) { m_time = time; }

		std::uint32_t GetAvailableStall() const { return m_stall; }
		void SetAvailableStall(std::uint32_t stall) { m_stall = stall; }

		std::uint32_t GetBarrierTime() const { return m_barrierTime; }
		void SetBarrierTime(std::uint32_t barrierTime) { m_barrierTime = barrierTime; }

		std::uint32_t GetBarrierStall() const { return m_barrierTime; }
		void SetBarrierStall(std::uint32_t barrierStall) { m_barrierTime = barrierStall; }

		std::uint32_t GetDependencyCount() const { return m_dependencyCount; }
		void SetDependencyCount(std::uint32_t dependencyCount) { m_dependencyCount = dependencyCount; }

	private:
		std::uint32_t m_time = 0;
		std::uint32_t m_stall = 0;
		std::uint32_t m_barrierTime = 0;
		std::uint32_t m_barrierStall = 0;

		std::uint32_t m_dependencyCount = 0;
	};

	robin_hood::unordered_map<SASS::Instruction *, InstructionProperties> instructionProperties;
	instructionProperties.reserve(instructionCount);

	// Maintain a list of barriered instructions, used to insert DEPBAR. Paired integer indicates
	// the position in the barrier list, used for partial barriers

	robin_hood::unordered_map<SASS::Instruction *, std::pair<SASS::Schedule::Barrier, std::uint8_t>> writeDependencyBarriers;
	robin_hood::unordered_map<SASS::Instruction *, std::pair<SASS::Schedule::Barrier, std::uint8_t>> readDependencyBarriers;

	writeDependencyBarriers.reserve(instructionCount);
	readDependencyBarriers.reserve(instructionCount);

	for (const auto& dependencyGraph : dependencyAnalysis.GetGraphs())
	{
		// Order the instructions based on the length of the longest path

		dependencyGraph->ReverseTopologicalOrderBFS([&](SASS::Instruction *instruction, SASS::Analysis::BlockDependencyGraph::Node& node)
		{
			auto latency = HardwareProperties::GetLatency(instruction);
			auto barrierLatency = HardwareProperties::GetBarrierLatency(instruction);
			auto readHold = HardwareProperties::GetReadHold(instruction);

			auto maxSuccessor = 0;
			for (auto& edge : node.GetOutgoingEdges())
			{
				auto successor = edge->GetEnd();
				auto successorValue = dependencyGraph->GetNodeValue(successor);

				for (auto dependency : edge->GetDependencies())
				{
					switch (dependency)
					{
						case SASS::Analysis::BlockDependencyGraph::DependencyKind::WriteRead:
						{
							successorValue += latency + barrierLatency;
							break;
						}
						case SASS::Analysis::BlockDependencyGraph::DependencyKind::WriteReadPredicate:
						{
							successorValue += latency;
							break;
						}
						case SASS::Analysis::BlockDependencyGraph::DependencyKind::WriteWrite:
						{
							successorValue += 1;
							break;
						}
						case SASS::Analysis::BlockDependencyGraph::DependencyKind::ReadWrite:
						{
							successorValue += 1 + readHold;
							break;
						}
					}
				}

				if (successorValue > maxSuccessor)
				{
					maxSuccessor = successorValue;
				}
			}

			node.SetValue(maxSuccessor);
		});

		// maintain the unit availability time, used for independent instructions which share
		// the same functional unit. Used as a hint to the instruction, as the GPU will insert stalls if needed

		robin_hood::unordered_map<HardwareProperties::FunctionalUnit, std::uint32_t> unitTime;

		// Construct a priority queue for available instructions (all dependencies scheduled)
		//
		// Priority queue comparator returns false if values in correct order (true to reorder).
		// i.e. if the left item comes AFTER the right

		auto priorityFunction = [&](SASS::Instruction *left, SASS::Instruction *right)
		{
			// Property 1: Stall count [lower]

			auto stallLeft = instructionProperties.at(left).GetBarrierStall();
			auto stallRight = instructionProperties.at(right).GetBarrierStall();

			if (stallLeft != stallRight)
			{
				return stallLeft > stallRight;
			}

			// Property 2: Overall/path latency [higher]

			auto pathLeft = dependencyGraph->GetNodeValue(left);
			auto pathRight = dependencyGraph->GetNodeValue(right);

			if (pathLeft != pathRight)
			{
				return pathLeft < pathRight;
			}

			// Property 3: Tie breaker (groups related instructions)
			
			return instructionOrder.at(left) > instructionOrder.at(right);
		};

		std::deque<SASS::Instruction *> availableInstructions;

		// Initialize available instructions

		for (auto& [instruction, node] : dependencyGraph->GetNodes())
		{
			auto count = node.GetInDegree();
			if (count == 0)
			{
				availableInstructions.push_back(instruction);
			}

			// Time, stall, barrier time, barrier stall, dependency count

			instructionProperties.emplace(
				std::piecewise_construct,
				std::forward_as_tuple(instruction),
				std::forward_as_tuple(0, 1, 0, 1, count)
			);
		}

		// Main schedule loop, maintain the current virtual cycle count to schedule/stall instructions

		std::uint32_t time = 0u;
		auto first = true;

		while (!availableInstructions.empty())
		{
			// Get the next instruction without sorting (priorities change between iterations)

			auto it = std::max_element(std::begin(availableInstructions), std::end(availableInstructions), priorityFunction);
			auto instruction = *it;
			availableInstructions.erase(it);

			// Get stall count to the previous instruction

			auto& schedule = instruction->GetSchedule();
			auto stall = instructionProperties.at(instruction).GetAvailableStall();

			// Add wait barriers for each predecessor that has not been waited

			robin_hood::unordered_set<SASS::Schedule::Barrier> instructionBarriers;
			robin_hood::unordered_set<SASS::Schedule::Barrier> partialBarriers;
			std::array<std::uint8_t, BARRIER_COUNT> waitBarriers{};

			if (first)
			{
				// Clear all previous barriers

				for (auto i = 0u; i < BARRIER_COUNT; ++i)
				{
					auto& wait = barrierWait[i];
					auto& count = barrierCount[i];
					if (count > wait)
					{
						instructionBarriers.insert(BARRIER_MAP[i]);
					}
					wait = 0;
					count = 0;
				}
			}

			for (auto& edge : dependencyGraph->GetIncomingEdges(instruction))
			{
				auto predecessor = edge->GetStart();
				for (auto& dependency : edge->GetDependencies())
				{
					switch (dependency)
					{
						case SASS::Analysis::BlockDependencyGraph::DependencyKind::WriteRead:
						case SASS::Analysis::BlockDependencyGraph::DependencyKind::WriteReadPredicate:
						case SASS::Analysis::BlockDependencyGraph::DependencyKind::WriteWrite:
						{
							if (auto w_it = writeDependencyBarriers.find(predecessor); w_it != writeDependencyBarriers.end())
							{
								// Check for any active wait barriers

								auto& [barrier, position] = w_it->second;

								if (position > barrierWait[BARRIER_UNMAP(barrier)])
								{
									auto& val = waitBarriers[BARRIER_UNMAP(barrier)];
									if (position > val)
									{
										val = position;
										partialBarriers.insert(barrier);
									}
								}
								writeDependencyBarriers.erase(w_it);

								// If waiting on a write barrier, the associated read barrier can
								// be cleared if it is still active

								if (auto r_it = readDependencyBarriers.find(predecessor); r_it != readDependencyBarriers.end())
								{
									auto& [barrier, position] = r_it->second;

									auto& wait = barrierWait[BARRIER_UNMAP(barrier)];
									if (position > wait)
									{
										wait = position;
									}
									readDependencyBarriers.erase(r_it);
								}
							}
							break;
						}
						case SASS::Analysis::BlockDependencyGraph::DependencyKind::ReadWrite:
						{
							if (auto r_it = readDependencyBarriers.find(predecessor); r_it != readDependencyBarriers.end())
							{
								// Check for any active wait barriers

								auto& [barrier, position] = r_it->second;
								if (position > barrierWait[BARRIER_UNMAP(barrier)])
								{
									auto& val = waitBarriers[BARRIER_UNMAP(barrier)];
									if (position > val)
									{
										val = position;
										partialBarriers.insert(barrier);
									}
								}
								readDependencyBarriers.erase(r_it);
							}
							break;
						}
					}
				}
			}
			
			// Insert instruction barriers, partial and full

			auto barrierStall = 1u;

			for (auto barrier : partialBarriers)
			{
				auto position = waitBarriers[BARRIER_UNMAP(barrier)];

				// Only some barriers may be processed out of order (certain memory ops)

				auto outOfOrder = false;
				if (optionCBarrier)
				{
					if (barrier == SASS::Schedule::Barrier::SB0 || barrier == SASS::Schedule::Barrier::SB1 ||
						barrier == SASS::Schedule::Barrier::SB4 || barrier == SASS::Schedule::Barrier::SB5)
					{
						outOfOrder = true;
					}
				}

				// A partial barrier is used when there are additional instructions queued after

				if (outOfOrder && position < barrierCount[BARRIER_UNMAP(barrier)])
				{
					// Shorten previous instruction stall (if possible)

					auto previousInstruction = scheduledInstructions.back();
					auto& previousSchedule = previousInstruction->GetSchedule();

					// Ensure 2 cycles for barrier used immediately

					if (barrierStall < 2)
					{
						auto readBarrier = previousSchedule.GetReadBarrier();
						auto writeBarrier = previousSchedule.GetWriteBarrier();

						// Check if this instruction waits on a barrier set in the previous instruction

						if (readBarrier == barrier || writeBarrier == barrier)
						{
							barrierStall = 2;
						}
					}

					// Set stall for current and previous instructions

					auto previousStall = previousSchedule.GetStall();
					previousSchedule.SetStall(barrierStall);

					if (previousSchedule.GetReuseCache().size() == 0)
					{
						previousSchedule.SetYield(barrierStall < 13);
					}

					// Convert to the instruction barrier type

					auto barrierI = GetInstructionBarrier(barrier);
					auto waitCount = barrierCount[BARRIER_UNMAP(barrier)] - position;

					// Insert barrier to wait until the instruction queue is ready

					auto barrierInstruction = new SASS::DEPBARInstruction(
						barrierI, new SASS::I8Immediate(waitCount), SASS::DEPBARInstruction::Flags::LE
					);
					auto& barrierSchedule = barrierInstruction->GetSchedule();

					// Adjust the stall count to ensure prior instruction finished

					auto barrierLatency = HardwareProperties::GetLatency(barrierInstruction);
					std::int32_t currentStall = previousStall - barrierStall;

					if (barrierLatency > currentStall)
					{
						currentStall = barrierLatency;
					}

					barrierSchedule.SetStall(currentStall);
					barrierSchedule.SetYield(currentStall < 13); // Higher stall counts cannot yield

					scheduledInstructions.push_back(barrierInstruction);

					// A barrier instruction must execute in its entirety

					time += barrierStall;

					stall -= barrierStall;
					if (barrierLatency > stall)
					{
						stall = barrierLatency;
					}
					barrierStall = barrierLatency;

					// Update the current wait position for the barrier

					barrierWait[BARRIER_UNMAP(barrier)] = position;
				}
				else
				{
					// Clear the entire barrier

					instructionBarriers.insert(barrier);
					barrierWait[BARRIER_UNMAP(barrier)] = barrierCount[BARRIER_UNMAP(barrier)];
				}
			}

			schedule.SetWaitBarriers(instructionBarriers);

			// Add new barriers to schedule for both read and write dependencies

			if (HardwareProperties::GetBarrierLatency(instruction) > 0)
			{
				// Select next free barrier resource for the instruction

				auto barrier = SASS::Schedule::Barrier::SB0;
				switch (instruction->GetInstructionClass())
				{
					case SASS::Instruction::InstructionClass::GlobalMemoryLoad:
					{
						barrier = SASS::Schedule::Barrier::SB0;
						break;
					}
					case SASS::Instruction::InstructionClass::SharedMemoryLoad:
					{
						barrier = SASS::Schedule::Barrier::SB1;
						break;
					}
					case SASS::Instruction::InstructionClass::S2R:
					case SASS::Instruction::InstructionClass::DoublePrecision:
					case SASS::Instruction::InstructionClass::SpecialFunction:
					{
						barrier = SASS::Schedule::Barrier::SB2;
						break;
					}
				}
				schedule.SetWriteBarrier(barrier);

				// Maintain barrier set

				writeDependencyBarriers.emplace(instruction, std::make_pair(barrier, ++barrierCount[BARRIER_UNMAP(barrier)]));
			}

			if (HardwareProperties::GetReadHold(instruction) > 0)
			{
				// Select the next free barrier resource for the instruction

				auto barrier = SASS::Schedule::Barrier::SB3;
				switch (instruction->GetInstructionClass())
				{
					case SASS::Instruction::InstructionClass::DoublePrecision:
					case SASS::Instruction::InstructionClass::SpecialFunction:
					{
						barrier = SASS::Schedule::Barrier::SB3;
						break;
					}
					case SASS::Instruction::InstructionClass::SharedMemoryLoad:
					case SASS::Instruction::InstructionClass::SharedMemoryStore:
					{
						barrier = SASS::Schedule::Barrier::SB4;
						break;
					}
					case SASS::Instruction::InstructionClass::GlobalMemoryLoad:
					case SASS::Instruction::InstructionClass::GlobalMemoryStore:
					{
						barrier = SASS::Schedule::Barrier::SB5;
						break;
					}
				}
				schedule.SetReadBarrier(barrier);

				// Maintain barrier set

				readDependencyBarriers.emplace(instruction, std::make_pair(barrier, ++barrierCount[BARRIER_UNMAP(barrier)]));
			}

			// Cap the stall by the maximum value. Legal since throughput is hardware regulated, and
			// stalls > 15 can only be caused by throuput (variable length is handled through barriers

			if (stall > 15)
			{
				stall = 15;
			}

			// Instruction latency is the maximum of:
			//  - Instruction latency
			//  - Previous instruction stall - current stall
			// We can therefore guarantee that both instructions are finished by the end
			// of the stall. Required property for the end of schedulable blocks

			if (!first)
			{
				auto previousInstruction = scheduledInstructions.back();
				auto& previousSchedule = previousInstruction->GetSchedule();

				// Ensure 2 cycles for barrier used immediately

				if (stall < 2)
				{
					auto readBarrier = previousSchedule.GetReadBarrier();
					auto writeBarrier = previousSchedule.GetWriteBarrier();

					// Check if this instruction waits on a barrier set in the previous instruction

					if (instructionBarriers.find(readBarrier) != instructionBarriers.end() ||
						instructionBarriers.find(writeBarrier) != instructionBarriers.end())
					{
						stall = 2;
					}
				}

				// Set stall for current and previous instructions

				auto previousStall = previousSchedule.GetStall();
				previousSchedule.SetStall(stall);

				if (previousSchedule.GetReuseCache().size() == 0) {
					previousSchedule.SetYield(stall < 13);
				}

				auto latency = HardwareProperties::GetLatency(instruction);
				std::int32_t currentStall = previousStall - stall;

				if (latency > currentStall)
				{
					currentStall = latency;
				}

				schedule.SetStall(currentStall);
				schedule.SetYield(currentStall < 13); // Higher stall counts cannot yield

				time += stall;
			}
			else
			{
				// For the first instruction, stall the entire pipeline depth (corrected later)

				auto latency = HardwareProperties::GetLatency(instruction);

				schedule.SetStall(latency);
				schedule.SetYield(latency < 13); // Higher stall counts cannot yield
			}

			scheduledInstructions.push_back(instruction);

			// Store the time the functional unit becomes free

			auto instructionUnit = HardwareProperties::GetFunctionalUnit(instruction);
			unitTime.insert_or_assign(instructionUnit, time + HardwareProperties::GetThroughputLatency(instruction));

			// Register reuse cache

			if (optionReuse && !first && stall > 0 && HardwareProperties::GetReuseFlags(instruction))
			{
				auto previousInstruction = scheduledInstructions[scheduledInstructions.size() - 2];
				if (HardwareProperties::GetReuseFlags(previousInstruction))
				{
					auto previousOperands = previousInstruction->GetSourceOperands();
					auto currentOperands = instruction->GetSourceOperands();

					// Iterate pairwise through source operands, checking for register matches that are not overwritten

					robin_hood::unordered_set<SASS::Schedule::ReuseCache> reuseCache;

					auto count = std::min(previousOperands.size(), currentOperands.size());
					for (auto i = 0u; i < count; ++i)
					{
						auto previousOperand = previousOperands[i];
						auto currentOperand = currentOperands[i];

						if (previousOperand == nullptr || currentOperand == nullptr)
						{
							continue;
						}

						if (previousOperand->GetKind() == SASS::Operand::Kind::Register &&
							currentOperand->GetKind() == SASS::Operand::Kind::Register)
						{
							auto previousRegister = static_cast<SASS::Register *>(previousOperand);
							auto currentRegister = static_cast<SASS::Register *>(currentOperand);

							auto previousValue = previousRegister->GetValue();
							auto currentValue = currentRegister->GetValue();

							if (previousValue == currentValue && previousValue != SASS::Register::ZeroIndex)
							{
								// Overwrites disable reuse opportunities

								auto overwritten = false;
								for (auto previousTarget : previousInstruction->GetDestinationOperands())
								{
									if (previousTarget == nullptr)
									{
										continue;
									}
									if (previousTarget->GetKind() == SASS::Operand::Kind::Register)
									{
										auto targetRegister = static_cast<SASS::Register *>(previousTarget);
										if (targetRegister->GetValue() == currentRegister->GetValue())
										{
											overwritten = true;
										}
									}
								}

								for (auto target : instruction->GetDestinationOperands())
								{
									if (target == nullptr)
									{
										continue;
									}

									if (target->GetKind() == SASS::Operand::Kind::Register)
									{
										auto targetRegister = static_cast<SASS::Register *>(target);
										if (targetRegister->GetValue() == currentRegister->GetValue())
										{
											overwritten = true;
										}
									}
								}

								if (overwritten)
								{
									continue;
								}

								// Translate from index to the operand cache entry

								const SASS::Schedule::ReuseCache CACHE_MAP[3] = {
									SASS::Schedule::ReuseCache::OperandA,
									SASS::Schedule::ReuseCache::OperandB,
									SASS::Schedule::ReuseCache::OperandC
								};
								reuseCache.insert(CACHE_MAP[i]);
							}
						}
					}

					// Set current reuse flags, and add reuse flags to previous instruction

					if (reuseCache.size() > 0)
					{
						schedule.SetReuseCache(reuseCache);
						schedule.SetYield(false); // May not yield and reuse

						auto& previousSchedule = previousInstruction->GetSchedule();
						auto& previousCache = previousSchedule.GetReuseCache();

						previousCache.insert(std::begin(reuseCache), std::end(reuseCache));
						previousSchedule.SetYield(false);
					}
				}
			}

			// Decrease the degree of all successors, adding them to the priority queue if next

			for (auto& edge : dependencyGraph->GetOutgoingEdges(instruction))
			{
				// Update the earliest time at which the instruction can be executed
				//  - Write/Read (true): instruction latency - read latency
				//  - Read/Write (anti): 1
				//  - Write/Write: 1
				// For instructions which have a barrier, delay 2 cycles (+1 for barrier)

				auto successor = edge->GetEnd();
				auto delay = 0u;
				auto barrierDelay = 0u;

				for (auto dependency : edge->GetDependencies())
				{
					switch (dependency)
					{
						case SASS::Analysis::BlockDependencyGraph::DependencyKind::WriteRead:
						{
							if (auto barrierLatency = HardwareProperties::GetBarrierLatency(instruction); barrierLatency > 0)
							{
								// If the previous instruction sets a barrier, minimum 2 cycle stall

								auto latency = 2;
								if (delay < latency)
								{
									delay = latency;
								}

								if (barrierDelay < barrierLatency)
								{
									barrierDelay = barrierLatency;
								}
							}
							else
							{
								// If no barrier, wait the full instruction length

								auto latency = HardwareProperties::GetLatency(instruction);
								auto readLatency = HardwareProperties::GetReadLatency(successor);

								auto diff = (int)latency - (int)readLatency;
								if (delay < diff)
								{
									delay = diff;
								}
							}
							break;
						}
						case SASS::Analysis::BlockDependencyGraph::DependencyKind::WriteReadPredicate:
						{
							// Predicate dependencies used for masking have no read latency

							auto latency = HardwareProperties::GetLatency(instruction);
							if (delay < latency)
							{
								delay = latency;
							}
							break;
						}

						// For both RAW/WAW, if the previous instruction sets a barrier, wait a minimum of 2 cycles.
						// Otherwise, must wait until the next cycle before executing

						case SASS::Analysis::BlockDependencyGraph::DependencyKind::ReadWrite:
						{
							if (auto barrierLatency = HardwareProperties::GetReadHold(instruction); barrierLatency > 0)
							{
								auto latency = 2;
								if (delay < latency)
								{
									delay = latency;
								}

								if (barrierDelay < barrierLatency)
								{
									barrierDelay = barrierLatency;
								}
							}
							else if (delay < 1)
							{
								delay = 1;
							}
							break;
						}
						case SASS::Analysis::BlockDependencyGraph::DependencyKind::WriteWrite:
						{
							if (auto barrierLatency = HardwareProperties::GetBarrierLatency(instruction); barrierLatency > 0)
							{
								auto latency = 2;
								if (delay < latency)
								{
									delay = latency;
								}

								if (barrierDelay < barrierLatency)
								{
									barrierDelay = barrierLatency;
								}
							}
							else if (delay < 1)
							{
								delay = 1;
							}
							break;
						}
					}
				}

				// Update instruction schedule properties

				auto& properties = instructionProperties.at(successor);

				// Record the latest schedulable time (dependends on all predecessors)

				auto availableTime = time + delay;

				if (availableTime > properties.GetAvailableTime())
				{
					properties.SetAvailableTime(availableTime);
				}

				// Record the latest schedulable time (dependends on all predecessors)
				// Also record the expected instruction availability (satisfying all barriers)

				if (barrierDelay < delay)
				{
					barrierDelay = delay;
				}

				auto barrierTime = time + barrierDelay;

				if (barrierTime > properties.GetBarrierTime())
				{
					properties.SetBarrierTime(barrierTime);
				}

				// Priority queue management

				auto dependencyCount = properties.GetDependencyCount() - 1;
				properties.SetDependencyCount(dependencyCount);

				if (dependencyCount == 0)
				{
					availableInstructions.push_back(successor);
				}
			}

			// Update the stall required for each available instruction

			for (auto availableInstruction : availableInstructions)
			{
				auto& properties = instructionProperties.at(availableInstruction);

				// Get time when the instruction dependencies are satisfied

				auto availableTime = properties.GetAvailableTime();

				// Check if the unit is busy with another (independent) instruction

				auto availableUnit = HardwareProperties::GetFunctionalUnit(availableInstruction);

				if (auto it = unitTime.find(availableUnit); it != unitTime.end())
				{
					auto unitAvailableTime = it->second;
					if (availableTime < unitAvailableTime)
					{
						availableTime = unitAvailableTime;
					}
				}

				// Compute the stall time

				std::int32_t stall = availableTime - time;

				// Require minimum stall count for some instructions (control)

				auto minimumStall = HardwareProperties::GetMinimumLatency(instruction);
				if (stall < minimumStall)
				{
					stall = minimumStall;
				}

				// Dual issue eligible instructions

				if (optionDual && availableTime <= time && HardwareProperties::GetDualIssue(instruction) &&
					availableUnit != HardwareProperties::GetFunctionalUnit(instruction))
				{
					auto constant = false;
					for (auto operand : instruction->GetSourceOperands())
					{
						if (operand->GetKind() == SASS::Operand::Kind::Constant)
						{
							constant = true;
						}
					}

					auto constantClash = false;
					if (constant)
					{
						for (auto operand : availableInstruction->GetSourceOperands())
						{
							if (operand->GetKind() == SASS::Operand::Kind::Constant)
							{
								constantClash = true;
							}
						}
					}

					// Cannot dual issue instructions which both load constants

					if (!constantClash)
					{
						// First instruction may be dual issued with second

						if (first)
						{
							stall = 0;
						}
						else
						{
							// Otherwise, make sure there was not already a dual issue in the previous pair

							auto previousInstruction = scheduledInstructions[scheduledInstructions.size() - 2];
							if (previousInstruction->GetSchedule().GetStall() != 0)
							{
								stall = 0;
							}
						}
					}
				}

				properties.SetAvailableStall(stall);

				// Hint for instruction execution

				auto barrierTime = properties.GetBarrierTime();

				std::int32_t barrierStall = barrierTime - time;
				if (barrierStall < stall)
				{
					barrierStall = stall;
				}

				properties.SetBarrierStall(barrierStall);
			}

			first = false;
		}
	}

	// For all active barriers at the end of the block, insert a barrier instruction

	for (auto i = 0u; i < BARRIER_COUNT; ++i)
	{
		// Only wait if active

		if (barrierWait[i] >= barrierCount[i])
		{
			continue;
		}

		// Convert to the instruction barrier type

		auto barrierI = GetInstructionBarrier(BARRIER_MAP[i]);

		// Insert barrier to wait until zero

		auto barrierInstruction = new SASS::DEPBARInstruction(
			barrierI, new SASS::I8Immediate(0x0), SASS::DEPBARInstruction::Flags::LE
		);
		scheduledInstructions.push_back(barrierInstruction);

		auto& barrierSchedule = barrierInstruction->GetSchedule();
		barrierSchedule.SetStall(HardwareProperties::GetLatency(barrierInstruction));
		barrierSchedule.SetYield(true);
	}

	Utils::Chrono::End(timeScheduler_start);
}

SASS::DEPBARInstruction::Barrier ListBlockScheduler::GetInstructionBarrier(SASS::Schedule::Barrier barrier) const
{
	switch (barrier)
	{
		case SASS::Schedule::Barrier::SB0:
		{
			return SASS::DEPBARInstruction::Barrier::SB0;
		}
		case SASS::Schedule::Barrier::SB1:
		{
			return SASS::DEPBARInstruction::Barrier::SB1;
		}
		case SASS::Schedule::Barrier::SB2:
		{
			return SASS::DEPBARInstruction::Barrier::SB2;
		}
		case SASS::Schedule::Barrier::SB3:
		{
			return SASS::DEPBARInstruction::Barrier::SB3;
		}
		case SASS::Schedule::Barrier::SB4:
		{
			return SASS::DEPBARInstruction::Barrier::SB4;
		}
		case SASS::Schedule::Barrier::SB5:
		{
			return SASS::DEPBARInstruction::Barrier::SB5;
		}
	}
	Utils::Logger::LogError("Unsupported barrier kind");
}

}
}
