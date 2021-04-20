#include "Backend/Scheduler/LinearBlockScheduler.h"

#include "Backend/Scheduler/HardwareProperties.h"

#include "Utils/Chrono.h"

namespace Backend {
namespace Scheduler {

void LinearBlockScheduler::ScheduleBlock(SASS::BasicBlock *block)
{
	auto timeScheduler_start = Utils::Chrono::Start("Linear scheduler '" + block->GetName() + "'");

	std::vector<SASS::Instruction *> scheduledInstructions;

	for (auto& instruction : block->GetInstructions())
	{
		scheduledInstructions.push_back(instruction);

		//TODO: 7 should be set to a constant somewhere for no barrier

		auto latency = HardwareProperties::GetLatency(instruction);
		auto barrierLatency = HardwareProperties::GetBarrierLatency(instruction);
		auto readHold = HardwareProperties::GetReadHold(instruction);

		if (barrierLatency > 0)
		{
			instruction->SetSchedule(
				latency, // Stall
				true,    // Yield
				0,       // Write barrier
				7,       // Read barrier
				0,       // Wait barriers
				0        // Reuse
			);

			scheduledInstructions.push_back(new SASS::DEPBARInstruction(
				SASS::DEPBARInstruction::Barrier::SB0, new SASS::I8Immediate(0x0), SASS::DEPBARInstruction::Flags::LE
			));
		}
		else if (readHold > 0)
		{
			instruction->SetSchedule(
				latency, // Stall
				true,    // Yield
				7,       // Write barrier
				0,       // Read barrier
				0,       // Wait barriers
				0        // Reuse
			);

			scheduledInstructions.push_back(new SASS::DEPBARInstruction(
				SASS::DEPBARInstruction::Barrier::SB0, new SASS::I8Immediate(0x0), SASS::DEPBARInstruction::Flags::LE
			));
		}
		else
		{
			instruction->SetSchedule(
				latency, // Stall
				true,    // Yield
				7,       // Write barrier
				7,       // Read barrier
				0,       // Wait barriers
				0        // Reuse
			);
		}
	}

	block->SetInstructions(scheduledInstructions);

	Utils::Chrono::End(timeScheduler_start);
}

}
}
