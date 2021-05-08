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

		auto latency = HardwareProperties::GetLatency(instruction);
		auto barrierLatency = HardwareProperties::GetBarrierLatency(instruction);
		auto readHold = HardwareProperties::GetReadHold(instruction);

		auto& schedule = instruction->GetSchedule();
		schedule.SetStall(latency);
		schedule.SetYield(true);
		schedule.SetReuseCache({SASS::Schedule::ReuseCache::OperandA});

		if (barrierLatency > 0 || readHold > 0)
		{
			if (barrierLatency > 0)
			{
				schedule.SetWriteBarrier(SASS::Schedule::Barrier::SB0);
			}
			else if (readHold > 0)
			{
				schedule.SetReadBarrier(SASS::Schedule::Barrier::SB0);
			}

			auto barrier = new SASS::DEPBARInstruction(
				SASS::DEPBARInstruction::Barrier::SB0, new SASS::I8Immediate(0x0), SASS::DEPBARInstruction::Flags::LE
			);
			auto& barrierSchedule = barrier->GetSchedule();
			barrierSchedule.SetStall(HardwareProperties::GetLatency(barrier));
			barrierSchedule.SetYield(true);

			scheduledInstructions.push_back(barrier);
		}
	}

	block->SetInstructions(scheduledInstructions);

	Utils::Chrono::End(timeScheduler_start);
}

}
}
