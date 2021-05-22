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

	SASS::Analysis::BlockDependencyAnalysis dependencyAnalysis;
	dependencyAnalysis.Build(block);

	auto timeScheduler_start = Utils::Chrono::Start("List scheduler '" + block->GetName() + "'");

	// Get scheduler options
	
	auto optionReuse = Utils::Options::IsBackendSchedule_Reuse();
	auto optionDual = Utils::Options::IsBackendSchedule_Dual();
	auto optionCBarrier = Utils::Options::IsBackendSchedule_CBarrier();

	// Build the schedule for each schedulable section individually (guarantees ordering)

	auto& scheduledInstructions = block->GetInstructions();
	auto instructionCount = scheduledInstructions.size();

	scheduledInstructions.clear();
	scheduledInstructions.reserve(instructionCount);

	for (const auto& dependencyGraph : dependencyAnalysis.GetGraphs())
	{
		// Order the instructions based on the length of the longest path

		robin_hood::unordered_map<SASS::Instruction *, std::uint32_t> topologicalOrder;
		auto orderValue = 0;

		dependencyGraph->ReverseTopologicalOrderBFS([&](
			// const SASS::Analysis::BlockDependencyGraph::OrderContextBFS& context, SASS::Instruction *instruction
			SASS::Instruction *instruction
		) {
			auto latency = HardwareProperties::GetLatency(instruction) +
				HardwareProperties::GetBarrierLatency(instruction) +
				HardwareProperties::GetReadHold(instruction);

			auto maxSuccessor = 0;

			for (auto edge : dependencyGraph->GetOutgoingEdges(instruction))
			{
				auto successor = edge->GetEnd();
				auto successorValue = dependencyGraph->GetNodeValue(successor);
				if (successorValue > maxSuccessor)
				{
					maxSuccessor = successorValue;
				}
			}

			dependencyGraph->SetNodeValue(instruction, latency + maxSuccessor);
			topologicalOrder.emplace(instruction, orderValue++);
			return true;
		});

		// Schedule the instructions based on the dependency DAG and hardware properties
		//   - Priority function: lowest stall count
		//   - Pipeline depth & latencies

		// For each instruction maintain:
		//   - (all) Earliest virtual clock cycle for execution (updated when each parent is scheduled)
		//   - (avail) Stall count required if scheduled next
		// Together these give the legal execution time for the instruction
		//
		// Also maintain the unit availability time, used for independent instructions which share
		// the same functional unit. Used as a hint to the instruction, as the GPU will insert stalls if needed
		//
		// For variable latency dependencies, we use a second map to hint at the expected execution time

		robin_hood::unordered_map<SASS::Instruction *, std::uint32_t> instructionTime;
		robin_hood::unordered_map<SASS::Instruction *, std::uint32_t> instructionStall;
		robin_hood::unordered_map<HardwareProperties::FunctionalUnit, std::uint32_t> unitTime;

		robin_hood::unordered_map<SASS::Instruction *, std::uint32_t> instructionBarrierTime;
		robin_hood::unordered_map<SASS::Instruction *, std::uint32_t> instructionBarrierStall;

		// Construct a priority queue for available instructions (all dependencies scheduled)
		//
		// Priority queue comparator returns false if values in correct order (true to reorder).
		// i.e. if the left item comes AFTER the right

		auto priorityFunction = [&](SASS::Instruction *left, SASS::Instruction *right) {
			// Property 1: Stall count [lower]

			auto stallLeft = instructionBarrierStall.at(left);
			auto stallRight = instructionBarrierStall.at(right);

			// auto stallLeft = instructionStall.at(left);
			// auto stallRight = instructionStall.at(right);
			if (stallLeft != stallRight)
			{
				return stallLeft > stallRight;
			}

			// Property 2: Individual latency [higher]

			// auto latencyLeft = HardwareProperties::GetLatency(left) +
			// 	HardwareProperties::GetBarrierLatency(left) +
			// 	HardwareProperties::GetReadHold(left);

			// auto latencyRight = HardwareProperties::GetLatency(right) +
			// 	HardwareProperties::GetBarrierLatency(right) +
			// 	HardwareProperties::GetReadHold(right);

			// if (latencyLeft != latencyRight)
			// {
			// 	// return latencyLeft < latencyRight;
			// }

			// Property 3: Overall/path latency [higher]

			auto pathLeft = dependencyGraph->GetNodeValue(left);
			auto pathRight = dependencyGraph->GetNodeValue(right);

			if (pathLeft != pathRight)
			{
				return pathLeft < pathRight;
			}

			// Property 4: Tie breaker
			
			return topologicalOrder.at(left) < topologicalOrder.at(right);
		};

		std::vector<SASS::Instruction *> availableInstructions;
		availableInstructions.reserve(dependencyGraph->GetNodeCount());

		// Initialize available instructions

		robin_hood::unordered_map<SASS::Instruction *, unsigned int> dependencyCount;

		for (auto& instruction : dependencyGraph->GetNodes())
		{
			auto count = dependencyGraph->GetInDegree(instruction);
			if (count == 0)
			{
				instructionTime.emplace(instruction, 0);
				instructionStall.emplace(instruction, 1);

				instructionBarrierTime.emplace(instruction, 0);
				instructionBarrierStall.emplace(instruction, 1);

				availableInstructions.push_back(instruction);
			}
			dependencyCount.emplace(instruction, count);
		}

		// For each barrier, keep track of the number of queued instructions, and the current wait position

		robin_hood::unordered_map<SASS::Schedule::Barrier, std::uint8_t> barrierCount;
		robin_hood::unordered_map<SASS::Schedule::Barrier, std::uint8_t> barrierWait;

		// Maintain a list of barriered instructions, used to insert DEPBAR. Paired integer indicates
		// the position in the barrier list, used for partial barriers

		robin_hood::unordered_map<SASS::Instruction *, std::pair<SASS::Schedule::Barrier, std::uint8_t>> writeDependencyBarriers;
		robin_hood::unordered_map<SASS::Instruction *, std::pair<SASS::Schedule::Barrier, std::uint8_t>> readDependencyBarriers;

		// Main schedule loop, maintain the current virtual cycle count to schedule/stall instructions

		std::uint32_t time = 0u;
		auto first = true;

		while (!availableInstructions.empty())
		{
			// Get the next instruction without sorting (priorities change between iterations)

			auto it = std::max_element(std::begin(availableInstructions), std::end(availableInstructions), priorityFunction);
			auto instruction = *it;
			availableInstructions.erase(it);

			auto& schedule = instruction->GetSchedule();

			// Get stall count to the previous instruction

			auto stall = instructionStall.at(instruction);
			instructionStall.erase(instruction);
			instructionTime.erase(instruction);

			// Add wait barriers for each predecessor that has not been waited

			robin_hood::unordered_map<SASS::Schedule::Barrier, std::uint8_t> waitBarriers;

			for (auto edge : dependencyGraph->GetIncomingEdges(instruction))
			{
				auto predecessor = edge->GetStart();
				for (auto dependency : edge->GetDependencies())
				{
					switch (dependency)
					{
						case SASS::Analysis::BlockDependencyGraph::DependencyKind::WriteRead:
						case SASS::Analysis::BlockDependencyGraph::DependencyKind::WriteWrite:
						{
							if (auto it = writeDependencyBarriers.find(predecessor); it != writeDependencyBarriers.end())
							{
								// Check for any active wait barriers

								auto& [barrier, position] = it->second;
								if (position > barrierWait[barrier])
								{
									if (position > waitBarriers[barrier])
									{
										waitBarriers.at(barrier) = position;
									}
								}
								writeDependencyBarriers.erase(it);

								// If waiting on a write barrier, the associated read barrier can
								// be cleared if it is still active

								if (auto it = readDependencyBarriers.find(predecessor); it != readDependencyBarriers.end())
								{
									auto& [barrier, position] = it->second;
									if (position > barrierWait[barrier])
									{
										barrierWait.at(barrier) = position;
									}
									readDependencyBarriers.erase(it);
								}
							}
							break;
						}
						case SASS::Analysis::BlockDependencyGraph::DependencyKind::ReadWrite:
						{
							if (auto it = readDependencyBarriers.find(predecessor); it != readDependencyBarriers.end())
							{
								// Check for any active wait barriers

								auto& [barrier, position] = it->second;
								if (position > barrierWait[barrier])
								{
									if (position > waitBarriers[barrier])
									{
										waitBarriers.at(barrier) = position;
									}
								}
								readDependencyBarriers.erase(it);
							}
							break;
						}
					}
				}
			}
			
			// Insert instruction barriers, partial and full

			robin_hood::unordered_set<SASS::Schedule::Barrier> instructionBarriers;
			auto barrierStall = 1u;

			for (const auto& [barrier, position] : waitBarriers)
			{
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

				if (outOfOrder && position < barrierCount.at(barrier))
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
					previousSchedule.SetYield(barrierStall < 13);

					// Convert to the instruction barrier type

					auto barrierI = GetInstructionBarrier(barrier);
					auto waitCount = barrierCount.at(barrier) - position;

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

					stall = barrierLatency;
					barrierStall = barrierLatency;

					// Update the current wait position for the barrier

					barrierWait.insert_or_assign(barrier, position);
				}
				else
				{
					// Clear the entire barrier

					instructionBarriers.insert(barrier);
					barrierWait.insert_or_assign(barrier, barrierCount.at(barrier));
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

				writeDependencyBarriers.emplace(instruction, std::make_pair(barrier, ++barrierCount[barrier]));
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

				readDependencyBarriers.emplace(instruction, std::make_pair(barrier, ++barrierCount[barrier]));
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
				previousSchedule.SetYield(stall < 13);

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

			//TODO: Reuse cache
			// instruction->SetReuseCache();

			// Decrease the degree of all successors, adding them to the priority queue if next

			for (auto edge : dependencyGraph->GetOutgoingEdges(instruction))
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

				// Record the latest schedulable time (dependends on all predecessors)
			       
				auto availableTime = time + delay;

				auto it = instructionTime.find(successor);
				if (it == instructionTime.end() || availableTime > it->second)
				{
					instructionTime.insert_or_assign(successor, availableTime);
				}

				// Also record the expected instruction availability (satisfying all barriers)

				auto barrierTime = time + barrierDelay;

                                auto it2 = instructionBarrierTime.find(successor);
				if (it2 == instructionBarrierTime.end() || barrierTime > it2->second)
				{
					instructionBarrierTime.insert_or_assign(successor, barrierTime);
				}

				// Priority queue management

				if (--dependencyCount.at(successor) == 0)
				{
					availableInstructions.push_back(successor);
				}
			}

			// Update the stall required for each available instruction

			for (auto availableInstruction : availableInstructions)
			{
				// Get time when the instruction dependencies are satisfied

				auto availableTime = instructionTime.at(availableInstruction);

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
					auto constant1 = false;
					for (auto operand : instruction->GetSourceOperands())
					{
						if (auto composite = dynamic_cast<SASS::Composite *>(operand))
						{
							if (composite->GetOpCodeKind() == SASS::Composite::OpCodeKind::Constant)
							{
								constant1 = true;
							}
						}
					}

					auto constant2 = false;
					for (auto operand : availableInstruction->GetSourceOperands())
					{
						if (auto composite = dynamic_cast<SASS::Composite *>(operand))
						{
							if (composite->GetOpCodeKind() == SASS::Composite::OpCodeKind::Constant)
							{
								constant2 = true;
							}
						}
					}

					// Cannot dual issue instructions which both load constants

					if (!constant1 || !constant2)
					{
						// First instruction may be dual issued with second

						if (first)
						{
							stall = 0;
						}
						else
						{
							// Otherwise, make sure there was not already a dual issue in the previous pair

							auto previousInstruction = scheduledInstructions.at(scheduledInstructions.size() - 2);
							if (previousInstruction->GetSchedule().GetStall() != 0)
							{
								stall = 0;
							}
						}
					}
				}

				instructionStall.insert_or_assign(availableInstruction, stall);

				// Hint for instruction execution

				auto barrierTime = instructionBarrierTime.at(availableInstruction);

				std::int32_t barrierStall = barrierTime - time;
				if (barrierStall < stall)
				{
					barrierStall = stall;
				}

				instructionBarrierStall.insert_or_assign(availableInstruction, barrierStall);
			}

			first = false;
		}

		// For all active barriers at the end of the block, insert a barrier instruction

		for (const auto& [barrier, count] : barrierCount)
		{
			// Only wait if active

			if (barrierWait[barrier] >= count)
			{
				continue;
			}

			// Convert to the instruction barrier type

			auto barrierI = GetInstructionBarrier(barrier);

			// Insert barrier to wait until zero

			auto barrierInstruction = new SASS::DEPBARInstruction(
				barrierI, new SASS::I8Immediate(0x0), SASS::DEPBARInstruction::Flags::LE
			);
			scheduledInstructions.push_back(barrierInstruction);

			auto& barrierSchedule = barrierInstruction->GetSchedule();
			barrierSchedule.SetStall(HardwareProperties::GetLatency(barrierInstruction));
			barrierSchedule.SetYield(true);
		}
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
