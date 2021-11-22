#include "Backend/Scheduler/LinearBlockScheduler.h"

#include "Backend/Scheduler/BarrierGenerator.h"

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

		auto latency = m_profile.GetLatency(instruction);
		auto barrierLatency = m_profile.GetBarrierLatency(instruction);
		auto readHold = m_profile.GetReadHold(instruction);

		auto& schedule = instruction->GetSchedule();
		schedule.SetStall(latency);
		schedule.SetYield(latency < 13);

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

			auto barrierInstruction = m_barrierGenerator.Generate(SASS::Schedule::Barrier::SB0);
			scheduledInstructions.push_back(barrierInstruction);
		}
	}

	block->SetInstructions(scheduledInstructions);

	Utils::Chrono::End(timeScheduler_start);
}

}
}
