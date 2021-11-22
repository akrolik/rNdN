#include "Backend/Scheduler/BarrierGenerator.h"

#include "Utils/Logger.h"

namespace Backend {
namespace Scheduler {

SASS::Instruction *BarrierGenerator::Generate(SASS::Schedule::Barrier barrier, std::uint8_t scoreboard, std::uint8_t latency) const
{
	auto computeCapability = m_profile.GetComputeCapability();
	if (SASS::Maxwell::IsSupported(computeCapability))
	{
		return Generate<SASS::Maxwell::DEPBARInstruction>(barrier, scoreboard, latency);
	}
	else if (SASS::Volta::IsSupported(computeCapability))
	{
		return Generate<SASS::Volta::DEPBARInstruction>(barrier, scoreboard, latency);
	}
	Utils::Logger::LogError("Unsupported CUDA compute capability for barrier 'sm_" + std::to_string(computeCapability) + "'");
}

template<class T>
T *BarrierGenerator::Generate(SASS::Schedule::Barrier barrier, std::uint8_t scoreboard, std::uint8_t latency) const
{
	// Create new barrier instruction

	auto barrierInstruction = new T(GetInstructionBarrier<T>(barrier), new SASS::I8Immediate(scoreboard), T::Flags::LE);

	// Get default latency for the instruction if needed

	if (latency == 0)
	{
		latency = m_profile.GetLatency(barrierInstruction);
	}

	// Update schedule

	auto& barrierSchedule = barrierInstruction->GetSchedule();
	barrierSchedule.SetStall(latency);
	barrierSchedule.SetYield(latency < 13); // Higher stall counts cannot yield
	
	return barrierInstruction;
}

template<class T>
typename T::Barrier BarrierGenerator::GetInstructionBarrier(SASS::Schedule::Barrier barrier) const
{
	switch (barrier)
	{
		case SASS::Schedule::Barrier::SB0:
		{
			return T::Barrier::SB0;
		}
		case SASS::Schedule::Barrier::SB1:
		{
			return T::Barrier::SB1;
		}
		case SASS::Schedule::Barrier::SB2:
		{
			return T::Barrier::SB2;
		}
		case SASS::Schedule::Barrier::SB3:
		{
			return T::Barrier::SB3;
		}
		case SASS::Schedule::Barrier::SB4:
		{
			return T::Barrier::SB4;
		}
		case SASS::Schedule::Barrier::SB5:
		{
			return T::Barrier::SB5;
		}
	}
	Utils::Logger::LogError("Unsupported barrier kind");
}

}
}
