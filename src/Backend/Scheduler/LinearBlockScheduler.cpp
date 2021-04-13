#include "Backend/Scheduler/LinearBlockScheduler.h"

#include "Backend/Scheduler/HardwareProperties.h"

namespace Backend {
namespace Scheduler {

void LinearBlockScheduler::ScheduleBlock(SASS::BasicBlock *block)
{
	for (auto& instruction : block->GetInstructions())
	{
		//TODO: 7 should be set to a constant somewhere for no barrier

		auto latency = HardwareProperties::GetLatency(instruction);
		auto barrierLatency = HardwareProperties::GetBarrierLatency(instruction);
		auto readHold = HardwareProperties::GetReadHold(instruction);

		if (barrierLatency > 0)
		{
			instruction->SetScheduling(
				latency, // Stall
				true,    // Yield
				0,       // Write barrier
				7,       // Read barrier
				0,       // Wait barriers
				0        // Reuse
			);
		}
		else if (readHold > 0)
		{
			instruction->SetScheduling(
				latency, // Stall
				true,    // Yield
				7,       // Write barrier
				0,       // Read barrier
				0,       // Wait barriers
				0        // Reuse
			);

		}
		else
		{
			instruction->SetScheduling(
				latency, // Stall
				true,    // Yield
				7,       // Write barrier
				7,       // Read barrier
				0,       // Wait barriers
				0        // Reuse
			);
		}
	}
}

}
}
