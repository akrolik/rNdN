#include "Backend/Scheduler/ListBlockScheduler.h"

#include "Backend/Scheduler/BarrierGenerator.h"

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

	// Build the schedule for each schedulable section individually (guarantees ordering)

	auto& scheduledInstructions = block->GetInstructions();
	auto instructionCount = scheduledInstructions.size();

	scheduledInstructions.clear();
	scheduledInstructions.reserve(instructionCount);

	// For each barrier, keep track of the number of queued instructions, and the current wait position

	std::array<std::uint8_t, HardwareProfile::BARRIER_COUNT> barrierCount{};
	std::array<std::uint8_t, HardwareProfile::BARRIER_COUNT> barrierWait{};

	// Schedule the instructions based on the dependency DAG and hardware properties:
	//   - Priority function: lowest stall count
	//   - Pipeline depth & latencies

	for (const auto& dependencyGraph : dependencyAnalysis.GetGraphs())
	{
		std::deque<SASS::Analysis::BlockDependencyGraph::Node *> availableInstructions;

		// Order the instructions based on the length of the longest path, and initialize available instructions

		dependencyGraph->ReverseTopologicalOrderBFS([&](SASS::Analysis::BlockDependencyGraph::Node& node)
		{
			auto instruction = node.GetInstruction();

			auto latency = m_profile.GetLatency(instruction);
			auto barrierLatency = m_profile.GetBarrierLatency(instruction);
			auto readHold = m_profile.GetReadHold(instruction);
			
			// Find the successor with the longest dependency chain

			auto maxSuccessor = 0;
			for (auto& edgeRef : node.GetOutgoingEdges())
			{
				const auto& edge = edgeRef.get();

				// Compute the expected delay before the next instruction (see stall computation below for details)

				auto maxDependency = 0;
				auto dependencies = edge.GetDependencies();

				if (dependencies & SASS::Analysis::BlockDependencyGraph::DependencyKind::WriteRead ||
					dependencies & SASS::Analysis::BlockDependencyGraph::DependencyKind::WriteReadPredicate ||
					dependencies & SASS::Analysis::BlockDependencyGraph::DependencyKind::WriteWrite)
				{
					if (barrierLatency > maxDependency)
					{
						maxDependency = barrierLatency;
					}
					else
					{
						auto successor = edge.GetEndInstruction();
						if (dependencies & SASS::Analysis::BlockDependencyGraph::DependencyKind::WriteRead ||
							dependencies & SASS::Analysis::BlockDependencyGraph::DependencyKind::WriteReadPredicate)
						{
							auto readLatency = m_profile.GetReadLatency(successor);
							if (dependencies & SASS::Analysis::BlockDependencyGraph::DependencyKind::WriteReadPredicate)
							{
								readLatency = 0;
							}

							auto diff = (int)latency - (int)readLatency;
							if (diff > maxDependency)
							{
								maxDependency = diff;
							}
						}

						if (dependencies & SASS::Analysis::BlockDependencyGraph::DependencyKind::WriteWrite)
						{
							auto throughput = m_profile.GetThroughputLatency(instruction);
							auto successorLatency = m_profile.GetLatency(successor);
							auto successorThroughput = m_profile.GetThroughputLatency(successor);

							auto length = latency - throughput;
							auto successorLength = successorLatency - successorThroughput;
							if (successorLength < 0)
							{
								successorLength = 0;
							}

							auto delay = 1;
							if (length > successorLength)
							{
								delay = (length - successorLength) + 1;
							}
							if (delay > maxDependency)
							{
								maxDependency = delay;
							}
						}
					}
				}

				if (dependencies & SASS::Analysis::BlockDependencyGraph::DependencyKind::ReadWrite)
				{
					if (readHold > maxDependency)
					{
						maxDependency = readHold;
					}
					else if (1 > maxDependency)
					{
						maxDependency = 1;
					}
				}

				// Record the max successor only

				const auto& endProperties = edge.GetEndNode().GetScheduleProperties();
				if (auto value = maxDependency + endProperties.GetHeuristic(); value > maxSuccessor)
				{
					maxSuccessor = value;
				}
			}

			node.GetScheduleProperties().SetHeuristic(maxSuccessor);

			// Initialize instruction properties and available instructions

			auto count = node.GetInDegree();
			if (count == 0)
			{
				availableInstructions.push_back(&node);
			}
			else
			{
				node.GetScheduleProperties().SetDependencyCount(count);
			}
		});

		// Maintain the unit availability time, used for independent instructions which share the same
		// functional unit. Used as a hint to the instruction, as the GPU will insert stalls if needed

		std::array<std::uint32_t, HardwareProfile::UNIT_COUNT> unitTime{};

		// Construct a priority queue for available instructions (all dependencies scheduled)
		//
		// Priority queue comparator returns false if values in correct order (true to reorder).
		// i.e. if the left item comes AFTER the right

		auto priorityFunction = [&](SASS::Analysis::BlockDependencyGraph::Node *left, SASS::Analysis::BlockDependencyGraph::Node *right)
		{
			auto& leftProperties = left->GetScheduleProperties();
			auto& rightProperties = right->GetScheduleProperties();

			// Property 1: Stall count [lower]

			auto stallLeft = leftProperties.GetBarrierStall();
			auto stallRight = rightProperties.GetBarrierStall();

			if (stallLeft != stallRight)
			{
				return stallLeft > stallRight;
			}

			// Property 2: Overall/path latency [higher]

			auto pathLeft = leftProperties.GetHeuristic();
			auto pathRight = rightProperties.GetHeuristic();

			if (pathLeft != pathRight)
			{
				return pathLeft < pathRight;
			}

			// Property 3: Tie breaker (groups related instructions)
			
			auto leftInstruction = left->GetInstruction();
			auto rightInstruction = right->GetInstruction();

			return leftInstruction->GetLineNumber() > rightInstruction->GetLineNumber();
		};

		// Main schedule loop, maintain the current virtual cycle count to schedule/stall instructions

		std::uint32_t time = 0u;
		auto first = true;

		while (!availableInstructions.empty())
		{
			// Get the next instruction without sorting (priorities change between iterations)

			auto it = std::max_element(std::begin(availableInstructions), std::end(availableInstructions), priorityFunction);
			auto instructionNode = *it;
			auto instruction = instructionNode->GetInstruction();

			availableInstructions.erase(it);

			// Get stall count to the previous instruction

			auto& schedule = instruction->GetSchedule();
			auto stall = instructionNode->GetScheduleProperties().GetAvailableStall();

			// Add wait barriers for each predecessor that has not been waited

			auto instructionBarriers = SASS::Schedule::Barrier::None;
			auto partialBarriers = SASS::Schedule::Barrier::None;
			std::array<std::uint8_t, HardwareProfile::BARRIER_COUNT> waitBarriers{};

			if (first)
			{
				// Clear all previous barriers at the beginning of the block

				for (auto i = 0u; i < HardwareProfile::BARRIER_COUNT; ++i)
				{
					auto& wait = barrierWait[i];
					auto& count = barrierCount[i];
					if (count > wait)
					{
						instructionBarriers |= SASS::Schedule::BarrierFromIndex(i);
					}
					wait = 0;
					count = 0;
				}
			}

			// Find all read/write barriers to the predecessors

			for (auto& edgeRef : instructionNode->GetIncomingEdges())
			{
				auto& edge = edgeRef.get();

				// Get edge dependencies

				auto predecessor = edge.GetStartInstruction();
				auto& predecessorProperties = edge.GetStartNode().GetScheduleProperties();

				auto dependencies = edge.GetDependencies();
				if (dependencies & SASS::Analysis::BlockDependencyGraph::DependencyKind::WriteRead ||
					dependencies & SASS::Analysis::BlockDependencyGraph::DependencyKind::WriteReadPredicate ||
					dependencies & SASS::Analysis::BlockDependencyGraph::DependencyKind::WriteWrite)
				{
					if (auto packedBarrier = predecessorProperties.GetWriteDependencyBarrier(); packedBarrier > 0)
					{
						// Check for any active wait barriers

						auto barrier = (packedBarrier & 0xff);
						auto position = (packedBarrier >> 8);

						if (position > barrierWait[barrier])
						{
							auto& wait = waitBarriers[barrier];
							if (position > wait)
							{
								wait = position;
								partialBarriers |= SASS::Schedule::BarrierFromIndex(barrier);
							}
						}
						predecessorProperties.SetWriteDependencyBarrier(0);

						// If waiting on a write barrier, the associated read barrier can be cleared if it is still active

						if (auto packedBarrier = predecessorProperties.GetReadDependencyBarrier(); packedBarrier > 0)
						{
							auto barrier = (packedBarrier & 0xff);
							auto position = (packedBarrier >> 8);

							auto& wait = barrierWait[barrier];
							if (position > wait)
							{
								wait = position;
							}
							predecessorProperties.SetReadDependencyBarrier(0);
						}
					}
				}
				else if (dependencies & SASS::Analysis::BlockDependencyGraph::DependencyKind::ReadWrite)
				{
					if (auto packedBarrier = predecessorProperties.GetReadDependencyBarrier(); packedBarrier > 0)
					{
						// Check for any active wait barriers

						auto barrier = (packedBarrier & 0xff);
						auto position = (packedBarrier >> 8);

						if (position > barrierWait[barrier])
						{
							auto& wait = waitBarriers[barrier];
							if (position > wait)
							{
								wait = position;
								partialBarriers |= SASS::Schedule::BarrierFromIndex(barrier);
							}
						}
						predecessorProperties.SetReadDependencyBarrier(0);
					}
				}                              
			}
			
			// Insert instruction barriers, partial and full

			auto barrierStall = 1u;

			for (auto i = 0u; i < HardwareProfile::BARRIER_COUNT; ++i)
			{
				auto barrier = SASS::Schedule::BarrierFromIndex(i);
				if (!(partialBarriers & barrier))
				{
					continue;
				}	

				auto position = waitBarriers[i];

				// Only some barriers may be processed out of order (certain memory ops)

				auto outOfOrder = optionCBarrier && m_profile.GetScoreboardBarrier(barrier);

				// A partial barrier is used when there are additional instructions queued after

				if (outOfOrder && position < barrierCount[i])
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

					if (previousSchedule.GetReuseCache() == SASS::Schedule::ReuseCache::None)
					{
						previousSchedule.SetYield(barrierStall < 12);
					}

					// Insert barrier to wait until the instruction queue is ready

					auto waitCount = barrierCount[i] - position;

					// Adjust the stall count to ensure prior instruction finished

					auto barrierInstruction = m_barrierGenerator.Generate(barrier, waitCount);
					auto barrierLatency = m_profile.GetLatency(barrierInstruction);

					std::int32_t currentStall = previousStall - barrierStall;
					if (barrierLatency > currentStall)
					{
						currentStall = barrierLatency;
					}

					auto& barrierSchedule = barrierInstruction->GetSchedule();
					barrierSchedule.SetStall(currentStall);
					barrierSchedule.SetYield(currentStall < 12); // Higher stall counts cannot yield

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

					barrierWait[i] = position;
				}
				else
				{
					// Clear the entire barrier

					instructionBarriers |= barrier;
					barrierWait[i] = barrierCount[i];
				}
			}

			schedule.SetWaitBarriers(instructionBarriers);

			// Add new barriers to schedule for both read and write dependencies

			if (m_profile.GetBarrierLatency(instruction) > 0)
			{
				// Select next free barrier resource for the instruction

				auto barrier = m_profile.GetWriteBarrier(instruction);
				schedule.SetWriteBarrier(barrier);

				// Maintain barrier set

				auto barrierIndex = SASS::Schedule::BarrierIndex(barrier);
				auto barrierPosition = ++barrierCount[barrierIndex];

				auto packedBarrier = static_cast<std::uint16_t>(barrierIndex) | (static_cast<std::uint16_t>(barrierPosition) << 8);
				instructionNode->GetScheduleProperties().SetWriteDependencyBarrier(packedBarrier);
			}

			if (m_profile.GetReadHold(instruction) > 0)
			{
				// Select the next free barrier resource for the instruction

				auto barrier = m_profile.GetReadBarrier(instruction);
				schedule.SetReadBarrier(barrier);

				// Maintain barrier set

				auto barrierIndex = SASS::Schedule::BarrierIndex(barrier);
				auto barrierPosition = ++barrierCount[barrierIndex];

				auto packedBarrier = static_cast<std::uint16_t>(barrierIndex) | (static_cast<std::uint16_t>(barrierPosition) << 8);
				instructionNode->GetScheduleProperties().SetReadDependencyBarrier(packedBarrier);
			}

			// Cap the stall by the maximum value. Legal since throughput is hardware regulated, and
			// stalls > 15 can only be caused by throughput (variable length is handled through barriers)

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

					if (!!(instructionBarriers & readBarrier) || !!(instructionBarriers & writeBarrier))
					{
						stall = 2;
					}
				}

				// Set stall for current and previous instructions

				auto previousStall = previousSchedule.GetStall();
				previousSchedule.SetStall(stall);

				if (previousSchedule.GetReuseCache() == SASS::Schedule::ReuseCache::None)
				{
					previousSchedule.SetYield(stall < 12);
				}

				auto latency = m_profile.GetLatency(instruction);
				std::int32_t currentStall = previousStall - stall;

				if (latency > currentStall)
				{
					currentStall = latency;
				}

				schedule.SetStall(currentStall);
				schedule.SetYield(currentStall < 12); // Higher stall counts cannot yield

				time += stall;
			}
			else
			{
				// For the first instruction, stall the entire pipeline depth (corrected later)

				auto latency = m_profile.GetLatency(instruction);

				schedule.SetStall(latency);
				schedule.SetYield(latency < 12); // Higher stall counts cannot yield
			}

			scheduledInstructions.push_back(instruction);

			// Store the time the functional unit becomes free

			auto instructionUnit = m_profile.GetFunctionalUnit(instruction);
			auto instructionUnitIndex = HardwareProfile::FunctionalUnitIndex(instructionUnit);

			auto unitAvailable = time + m_profile.GetThroughputLatency(instruction);
			unitTime[instructionUnitIndex] = unitAvailable;

			// Register reuse cache

			if (optionReuse && !first && stall > 0 && m_profile.GetReuseFlags(instruction))
			{
				auto previousInstruction = scheduledInstructions[scheduledInstructions.size() - 2];
				if (m_profile.GetReuseFlags(previousInstruction))
				{
					auto previousOperands = previousInstruction->GetCachedSourceOperands();
					auto currentOperands = instruction->GetCachedSourceOperands();

					// Iterate pairwise through source operands, checking for register matches that are not overwritten

					auto reuseCache = SASS::Schedule::ReuseCache::None;

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
								for (auto previousTarget : previousInstruction->GetCachedDestinationOperands())
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

								for (auto target : instruction->GetCachedDestinationOperands())
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

								reuseCache |= SASS::Schedule::ReuseCacheFromIndex(i);
							}
						}
					}

					// Set current reuse flags, and add reuse flags to previous instruction

					if (reuseCache != SASS::Schedule::ReuseCache::None)
					{
						schedule.SetReuseCache(reuseCache);
						schedule.SetYield(false); // May not yield and reuse

						auto& previousSchedule = previousInstruction->GetSchedule();
						auto& previousCache = previousSchedule.GetReuseCache();

						previousCache |= reuseCache;
						previousSchedule.SetYield(false);
					}
				}
			}

			// Decrease the degree of all successors, adding them to the priority queue if next

			for (auto& edgeRef : instructionNode->GetOutgoingEdges())
			{
				auto& edge = edgeRef.get();

				// Update the earliest time at which the instruction can be executed
				//  - Write/Read (true): instruction latency - read latency
				//  - Read/Write (anti): 1
				//  - Write/Write: latency to maintain order
				// For instructions which have a barrier, delay 2 cycles (+1 for barrier)

				auto successor = edge.GetEndInstruction();
				auto& successorNode = edge.GetEndNode();

				auto delay = 0u;
				auto barrierDelay = 0u;

				auto dependencies = edge.GetDependencies();
				if (dependencies & SASS::Analysis::BlockDependencyGraph::DependencyKind::WriteRead ||
					dependencies & SASS::Analysis::BlockDependencyGraph::DependencyKind::WriteReadPredicate ||
					dependencies & SASS::Analysis::BlockDependencyGraph::DependencyKind::WriteWrite)
				{
					// For both RAW/WAW, if the previous instruction sets a barrier, wait a minimum of 2 cycles.

					if (auto barrierLatency = m_profile.GetBarrierLatency(instruction); barrierLatency > 0)
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
					else
					{
						// For WAW, need to maintain ordering, for RAW, ensure the value is written

						if (dependencies & SASS::Analysis::BlockDependencyGraph::DependencyKind::WriteReadPredicate ||
							dependencies & SASS::Analysis::BlockDependencyGraph::DependencyKind::WriteRead)
						{
							// If no barrier, wait the full instruction length

							auto latency = m_profile.GetLatency(instruction);
							auto readLatency = m_profile.GetReadLatency(successor);

							// Predicate dependencies used for masking have no read latency

							if (dependencies & SASS::Analysis::BlockDependencyGraph::DependencyKind::WriteReadPredicate)
							{
								readLatency = 0;
							}

							auto diff = (int)latency - (int)readLatency;
							if (delay < diff)
							{
								delay = diff;
							}
						}

						if (dependencies & SASS::Analysis::BlockDependencyGraph::DependencyKind::WriteWrite)
						{
							// Must wait so that it completes one cycle after the current instruction

							auto instructionLatency = m_profile.GetLatency(instruction);
							auto instructionThroughput = m_profile.GetThroughputLatency(instruction);

							auto successorLatency = m_profile.GetLatency(successor);
							auto successorThroughput = m_profile.GetThroughputLatency(successor);

							auto instructionLength = instructionLatency - instructionThroughput;
							auto successorLength = successorLatency - successorThroughput;
							if (successorLength < 0)
							{
								successorLength = 0;
							}

							auto writeDelay = 1;
							if (instructionLength > successorLength)
							{
								writeDelay = (instructionLength - successorLength) + 1;
							}
							if (delay < writeDelay)
							{
								delay = writeDelay;
							}
						}
					}
				}

				// WAR requires the previous value has already been read

				if (dependencies & SASS::Analysis::BlockDependencyGraph::DependencyKind::ReadWrite)
				{
					// If the instruction sets a read barrier, delay 2 cycles to become active

					if (auto barrierLatency = m_profile.GetReadHold(instruction); barrierLatency > 0)
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
						// Must wait until the next cycle before executing, we assume that the read latency
						// is small enough on all instructions to be irrelevant

						delay = 1;
					}
				}

				// Update instruction schedule properties

				auto& properties = successorNode.GetScheduleProperties();

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
					availableInstructions.push_back(&successorNode);
				}
			}

			// Update the stall required for each available instruction

			for (auto availableNode : availableInstructions)
			{
				auto availableInstruction = availableNode->GetInstruction();
				auto& properties = availableNode->GetScheduleProperties();

				// Get time when the instruction dependencies are satisfied

				auto availableTime = properties.GetAvailableTime();
				auto availableUnit = m_profile.GetFunctionalUnit(availableInstruction);
				auto availableUnitIndex = HardwareProfile::FunctionalUnitIndex(availableUnit);

				// Compute the stall time

				std::int32_t stall = (int)availableTime - (int)time;

				// Require minimum stall count for some instructions (control)

				auto minimumStall = m_profile.GetMinimumLatency(instruction);
				if (stall < minimumStall)
				{
					stall = minimumStall;
				}

				// Dual issue eligible instructions

				if (optionDual && availableTime <= time && m_profile.GetDualIssue(instruction) &&
					availableUnit != m_profile.GetFunctionalUnit(instruction))
				{
					auto constant = false;
					for (auto operand : instruction->GetCachedSourceOperands())
					{
						if (operand->GetKind() == SASS::Operand::Kind::Constant)
						{
							constant = true;
							break;
						}
					}

					auto constantClash = false;
					if (constant)
					{
						for (auto operand : availableInstruction->GetCachedSourceOperands())
						{
							if (operand->GetKind() == SASS::Operand::Kind::Constant)
							{
								constantClash = true;
								break;
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

				// Check if the unit is busy with another (independent) instruction

				if (auto unitAvailableTime = unitTime[availableUnitIndex]; barrierTime < unitAvailableTime)
				{
					barrierTime = unitAvailableTime;
				}

				// Compute expected stall

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

	for (auto i = 0u; i < HardwareProfile::BARRIER_COUNT; ++i)
	{
		// Only wait if active

		if (barrierWait[i] >= barrierCount[i])
		{
			continue;
		}

		// Insert barrier to wait until zero

		auto barrier = SASS::Schedule::BarrierFromIndex(i);
		auto barrierInstruction = m_barrierGenerator.Generate(barrier);

		scheduledInstructions.push_back(barrierInstruction);
	}

	Utils::Chrono::End(timeScheduler_start);
}

}
}
