#include "Backend/Scheduler/ListBlockScheduler.h"

#include "Backend/Scheduler/HardwareProperties.h"

#include "SASS/Analysis/Dependency/BlockDependencyAnalysis.h"

#include <queue>

#include "Utils/Chrono.h"

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
	
	// Build the schedule for each schedulable section individually (guarantees ordering)

	auto& scheduledInstructions = block->GetInstructions();
	scheduledInstructions.clear();

	for (const auto& dependencyGraph : dependencyAnalysis.GetGraphs())
	{
		// Order the instructions based on the length of the longest path

		dependencyGraph->ReverseTopologicalOrderDFS([&](const SASS::Analysis::BlockDependencyGraph::OrderContextDFS& context, SASS::Instruction *instruction)
		{
			auto latency = HardwareProperties::GetLatency(instruction) +
				HardwareProperties::GetBarrierLatency(instruction) +
				HardwareProperties::GetReadHold(instruction);

			auto maxSuccessor = 0;

			for (auto successor : dependencyGraph->GetSuccessors(instruction))
			{
				auto successorValue = dependencyGraph->GetNodeValue(successor);
				if (successorValue > maxSuccessor)
				{
					maxSuccessor = successorValue;
				}
			}

			dependencyGraph->SetNodeValue(instruction, latency + maxSuccessor);
			return true;
		});

		// Schedule the instructions based on the dependency DAG and hardware properties
		//   - Priority function: lowest stall count
		//   - Pipeline depth & latencies

		// For each instruction maintain:
		//   - (all) Earliest virtual clock cycle for execution (updated when each parent is scheduled)
		//   - (avail) Stall count required if scheduled next

		robin_hood::unordered_map<SASS::Instruction *, std::uint32_t> scheduleTime;
		robin_hood::unordered_map<SASS::Instruction *, std::uint32_t> scheduleStall;

		// Construct a priority queue for available instructions (all dependencies scheduled)
		// 
		// Priority queue comparator returns false if values in correct order (true to reorder)

		auto priorityFunction = [&](SASS::Instruction *left, SASS::Instruction *right) {
			return dependencyGraph->GetNodeValue(left) < dependencyGraph->GetNodeValue(right);
		};
		std::priority_queue<
			SASS::Instruction *, std::vector<SASS::Instruction *>, decltype(priorityFunction)
		> availableInstructions(priorityFunction);

		// Initialize available instructions

		robin_hood::unordered_map<SASS::Instruction *, unsigned int> dependencyCount;

		for (auto& instruction : dependencyGraph->GetNodes())
		{
			auto count = dependencyGraph->GetInDegree(instruction);
			if (count == 0)
			{
				scheduleTime.emplace(instruction, 0);
				scheduleStall.emplace(instruction, 0);

				availableInstructions.push(instruction);
			}
			dependencyCount.emplace(instruction, count);
		}

		// Maintain a list of barriered instructions, used to insert DEPBAR

		robin_hood::unordered_map<SASS::Instruction *, SASS::Schedule::Barrier> dependencyBarriers;
		robin_hood::unordered_set<SASS::Schedule::Barrier> activeBarriers;

		// Main schedule loop, maintain the current virtual cycle count to schedule/stall instructions

		std::uint32_t time = 0u;
		auto first = true;

		while (!availableInstructions.empty())
		{
			// Get the next instruction and stall count (to the previous instruction)

			auto instruction = availableInstructions.top();
			availableInstructions.pop();

			auto stall = scheduleStall.at(instruction);
			scheduleStall.erase(instruction);
			scheduleTime.erase(instruction);

			if (stall > 15)
			{
				Utils::Logger::LogError("Stall count exceeds maximum value [" + std::to_string(stall) + " > 15]");
			}

			// Schedule instruction

			scheduledInstructions.push_back(instruction);

			auto& schedule = instruction->GetSchedule();
			auto latency = HardwareProperties::GetLatency(instruction);

			// Instruction latency is the maximum of:
			//  - Instruction latency
			//  - Previous instruction stall - current stall
			// We can therefore guarantee that both instructions are finished by the end
			// of the stall. Required property for the end of schedulable blocks

			if (!first)
			{
				auto previousInstruction = scheduledInstructions.at(scheduledInstructions.size() - 2);
				auto& previousSchedule = previousInstruction->GetSchedule();

				auto previousStall = previousSchedule.GetStall();
				previousSchedule.SetStall(stall);
				previousSchedule.SetYield(stall < 13);

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

				schedule.SetStall(latency);
				schedule.SetYield(latency < 13); // Higher stall counts cannot yield
			}

			// Add wait barriers

			robin_hood::unordered_set<SASS::Schedule::Barrier> waitBarriers;

			for (auto predecessor : dependencyGraph->GetPredecessors(instruction))
			{
				auto it = dependencyBarriers.find(predecessor);
				if (it != dependencyBarriers.end())
				{
					waitBarriers.insert(it->second);
					dependencyBarriers.erase(it);
				}
			}

			schedule.SetWaitBarriers(waitBarriers);

			// Clear old barriers

			for (auto barrier : waitBarriers)
			{
				activeBarriers.erase(barrier);
			}

			// Add new barriers to schedule

			auto barrierLatency = HardwareProperties::GetBarrierLatency(instruction);
			auto readHold = HardwareProperties::GetReadHold(instruction);

			if (barrierLatency > 0 || readHold > 0)
			{
				// Instructions which set dependencies require 1 cycle to issue and 1 cycle to set barrier

				schedule.SetStall(2);

				// Set the read/write barrier, select from non-active set

				auto barrier = SASS::Schedule::Barrier::SB0;
				for (auto ba : {SASS::Schedule::Barrier::SB0, SASS::Schedule::Barrier::SB1,
						SASS::Schedule::Barrier::SB2, SASS::Schedule::Barrier::SB3,
						SASS::Schedule::Barrier::SB4, SASS::Schedule::Barrier::SB5})
				{
					if (activeBarriers.find(ba) == activeBarriers.end())
					{
						barrier = ba;
						break;
					}
				}

				if (barrierLatency > 0)
				{
					schedule.SetWriteBarrier(barrier);
				}
				else if (readHold > 0)
				{
					schedule.SetReadBarrier(barrier);
				}

				dependencyBarriers[instruction] = barrier;
				activeBarriers.insert(barrier);
			}

			//TODO: Reuse cache
			// instruction->SetReuseCache();

			// Decrease the degree of all successors, adding them to the priority queue if next

			std::vector<SASS::Instruction *> availableDependencies;
			for (auto& successor : dependencyGraph->GetSuccessors(instruction))
			{
				// Update the earliest time at which the instruction can be executed
				//  - Write/Read (true): instruction latency - read latency
				//  - Read/Write (anti): 1
				//  - Write/Write: 1

				auto delay = 0u;

				for (auto dependency : dependencyGraph->GetEdgeDependencies(instruction, successor))
				{
					switch (dependency)
					{
						case SASS::Analysis::BlockDependencyGraph::DependencyKind::WriteRead:
						{
							auto latency = HardwareProperties::GetLatency(instruction);
							auto readLatency = HardwareProperties::GetReadLatency(successor);

							auto diff = (int)latency - (int)readLatency;
							if (delay < diff)
							{
								delay = diff;
							}
							break;
						}
						case SASS::Analysis::BlockDependencyGraph::DependencyKind::ReadWrite:
						case SASS::Analysis::BlockDependencyGraph::DependencyKind::WriteWrite:
						{
							if (delay < 1)
							{
								delay = 1;
							}
							break;
						}
					}
				}

				auto availableTime = time + delay;

				// Record the latest schedulable time (dependends on all predecessors)
			       
				auto it = scheduleTime.find(successor);
				if (it == scheduleTime.end() || availableTime > it->second)
				{
					scheduleTime[successor] = availableTime;
				}

				// Priority queue management

				dependencyCount.at(successor)--;
				if (dependencyCount.at(successor) == 0)
				{
					scheduleStall[successor] = 0;
					availableDependencies.push_back(successor);
				}
			}

			// Update the stall required for each available instruction

			for (auto& [availableInstruction, _] : scheduleStall)
			{
				std::int32_t stall = scheduleTime.at(availableInstruction) - time;

				// Require minimum stall count for some instructions

				auto minimumStall = HardwareProperties::GetMinimumLatency(instruction);
				if (stall < minimumStall)
				{
					stall = minimumStall;
				}

				// Stall necessary when 2 instructions of the same class execute are back-to-back

				if (instruction->GetHardwareClass() == availableInstruction->GetHardwareClass())
				{
					auto throughput = HardwareProperties::GetThroughputLatency(instruction);
					if (stall < throughput)
					{
						stall = throughput;
					}
					
					//TODO: 16 used for DP unit
					if (stall > 15)
					{
						stall = 15;
					}
				}

				//TODO: Dual issue (no 2-constants)

				scheduleStall.at(availableInstruction) = stall;
			}

			// Add last to ensure priority queue ordering
			
			for (auto availableInstruction : availableDependencies)
			{
				availableInstructions.push(availableInstruction);
			}

			first = false;
		}

		// For all active barriers at the end of the block, insert a barrier instruction

		for (auto barrier : activeBarriers)
		{
			// Convert to the instruction barrier type

			SASS::DEPBARInstruction::Barrier barrierI;
			switch (barrier)
			{
				case SASS::Schedule::Barrier::SB0:
				{
					barrierI = SASS::DEPBARInstruction::Barrier::SB0;
					break;
				}
				case SASS::Schedule::Barrier::SB1:
				{
					barrierI = SASS::DEPBARInstruction::Barrier::SB1;
					break;
				}
				case SASS::Schedule::Barrier::SB2:
				{
					barrierI = SASS::DEPBARInstruction::Barrier::SB2;
					break;
				}
				case SASS::Schedule::Barrier::SB3:
				{
					barrierI = SASS::DEPBARInstruction::Barrier::SB3;
					break;
				}
				case SASS::Schedule::Barrier::SB4:
				{
					barrierI = SASS::DEPBARInstruction::Barrier::SB4;
					break;
				}
				case SASS::Schedule::Barrier::SB5:
				{
					barrierI = SASS::DEPBARInstruction::Barrier::SB5;
					break;
				}
			}

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

}
}
