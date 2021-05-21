#include "Backend/Scheduler/ListBlockScheduler.h"

#include "Backend/Scheduler/HardwareProperties.h"

#include "SASS/Analysis/Dependency/BlockDependencyAnalysis.h"

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

		// Maintain a list of barriered instructions, used to insert DEPBAR

		robin_hood::unordered_map<SASS::Instruction *, SASS::Schedule::Barrier> writeDependencyBarriers;
		robin_hood::unordered_map<SASS::Instruction *, SASS::Schedule::Barrier> readDependencyBarriers;

		robin_hood::unordered_set<SASS::Schedule::Barrier> activeBarriers;

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

			auto stall = instructionStall.at(instruction);
			instructionStall.erase(instruction);
			instructionTime.erase(instruction);

			// Cap the stall by the maximum value. Legal since throughput is hardware regulated, and
			// stalls > 15 can only be caused by throuput (variable length is handled through barriers

			if (stall > 15)
			{
				stall = 15;
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

			// Store the time the functional unit becomes free

			auto instructionUnit = HardwareProperties::GetFunctionalUnit(instruction);
			unitTime.insert_or_assign(instructionUnit, time + HardwareProperties::GetThroughputLatency(instruction));

			// Add wait barriers for each predecessor that has not been waited

			robin_hood::unordered_set<SASS::Schedule::Barrier> waitBarriers;

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
								auto barrier = it->second;
								if (activeBarriers.find(barrier) != activeBarriers.end())
								{
									waitBarriers.insert(barrier);
								}
								writeDependencyBarriers.erase(it);

								// If waiting on a write barrier, the associated read barrier can
								// be cleared if it is still active

								if (auto it2 = readDependencyBarriers.find(predecessor); it2 != readDependencyBarriers.end())
								{
									auto &schedule = predecessor->GetSchedule();
									schedule.SetReadBarrier(SASS::Schedule::Barrier::None);

									activeBarriers.erase(it2->second);
									readDependencyBarriers.erase(it2);
								}
							}
							break;
						}
						case SASS::Analysis::BlockDependencyGraph::DependencyKind::ReadWrite:
						{
							if (auto it = readDependencyBarriers.find(predecessor); it != readDependencyBarriers.end())
							{
								auto barrier = it->second;
								if (activeBarriers.find(barrier) != activeBarriers.end())
								{
									waitBarriers.insert(barrier);
								}
								readDependencyBarriers.erase(it);
							}
							break;
						}
					}
				}
			}

			schedule.SetWaitBarriers(waitBarriers);
			
			// Ensure 2 cycles for barrier used immediately

			if (!first)
			{
				auto previousInstruction = scheduledInstructions.at(scheduledInstructions.size() - 2);
				auto& previousSchedule = previousInstruction->GetSchedule();

				auto readBarrier = previousSchedule.GetReadBarrier();
				auto writeBarrier = previousSchedule.GetWriteBarrier();

				// Check if this instruction waits on a barrier set in the previous instruction

				if (waitBarriers.find(readBarrier) != waitBarriers.end() ||
					waitBarriers.find(writeBarrier) != waitBarriers.end())
				{
					// Minimum 2 cycles to issue instruction and set barrier active

					auto previousStall = previousSchedule.GetStall();
					auto minimumStall = 2;

					if (previousStall < minimumStall)
					{
						// Update stall count

						previousSchedule.SetStall(minimumStall);

						auto diff = minimumStall - previousStall;
						auto currentStall = schedule.GetStall() - diff;

						if (latency > currentStall)
						{
							currentStall = latency;
						}

						schedule.SetStall(currentStall);
						schedule.SetYield(currentStall < 13); // Higher stall counts cannot yield

						time += diff;
					}
				}
			}

			// Clear old barriers

			for (auto barrier : waitBarriers)
			{
				activeBarriers.erase(barrier);
			}

			// Add new barriers to schedule for both read and write dependencies

			//TODO: Barrier allocation scheme

			if (HardwareProperties::GetBarrierLatency(instruction) > 0)
			{
				// Select next free barrier resource for the instruction

				auto barrier = SASS::Schedule::Barrier::SB0;
				switch (instruction->GetInstructionClass())
				{
					case SASS::Instruction::InstructionClass::S2R:
					{
						barrier = SASS::Schedule::Barrier::SB1;
						break;
					}
					case SASS::Instruction::InstructionClass::DoublePrecision:
					{
						barrier = SASS::Schedule::Barrier::SB2;
						break;
					}
					case SASS::Instruction::InstructionClass::SpecialFunction:
					{
						barrier = SASS::Schedule::Barrier::SB3;
						break;
					}
					case SASS::Instruction::InstructionClass::SharedMemoryLoad:
					case SASS::Instruction::InstructionClass::GlobalMemoryLoad:
					{
						barrier = SASS::Schedule::Barrier::SB0;
						break;
					}
				}

				// auto barrier = SASS::Schedule::Barrier::SB0;
				// for (auto ba : {SASS::Schedule::Barrier::SB0, SASS::Schedule::Barrier::SB1,
				// 		SASS::Schedule::Barrier::SB2, SASS::Schedule::Barrier::SB3,
				// 		SASS::Schedule::Barrier::SB4, SASS::Schedule::Barrier::SB5})
				// {
				// 	if (activeBarriers.find(ba) == activeBarriers.end())
				// 	{
				// 		barrier = ba;
				// 		break;
				// 	}
				// }

				schedule.SetWriteBarrier(barrier);

				// Maintain barrier set

				writeDependencyBarriers.emplace(instruction, barrier);
				activeBarriers.insert(barrier);
			}

			if (HardwareProperties::GetReadHold(instruction) > 0)
			{
				// Select the next free barrier resource for the instruction

				auto barrier = SASS::Schedule::Barrier::SB4;
				switch (instruction->GetInstructionClass())
				{
					case SASS::Instruction::InstructionClass::DoublePrecision:
					case SASS::Instruction::InstructionClass::SpecialFunction:
					{
						barrier = SASS::Schedule::Barrier::SB4;
						break;
					}
					case SASS::Instruction::InstructionClass::SharedMemoryLoad:
					case SASS::Instruction::InstructionClass::SharedMemoryStore:
					case SASS::Instruction::InstructionClass::GlobalMemoryLoad:
					case SASS::Instruction::InstructionClass::GlobalMemoryStore:
					{
						barrier = SASS::Schedule::Barrier::SB5;
						break;
					}
				}
				// for (auto ba : {SASS::Schedule::Barrier::SB0, SASS::Schedule::Barrier::SB1,
				// 		SASS::Schedule::Barrier::SB2, SASS::Schedule::Barrier::SB3,
				// 		SASS::Schedule::Barrier::SB4, SASS::Schedule::Barrier::SB5})
				// {
				// 	if (activeBarriers.find(ba) == activeBarriers.end())
				// 	{
				// 		barrier = ba;
				// 		break;
				// 	}
				// }

				schedule.SetReadBarrier(barrier);

				// Maintain barrier set

				readDependencyBarriers.emplace(instruction, barrier);
				activeBarriers.insert(barrier);
			}

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

				auto instructionUnit = HardwareProperties::GetFunctionalUnit(availableInstruction);

				if (auto it = unitTime.find(instructionUnit); it != unitTime.end())
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

				//TODO: Dual issue (no 2-constants)

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
