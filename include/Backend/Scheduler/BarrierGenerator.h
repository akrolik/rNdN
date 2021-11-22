#pragma once

#include "Backend/Scheduler/Profiles/HardwareProfile.h"

#include "SASS/Tree/Tree.h"

namespace Backend {
namespace Scheduler {

class BarrierGenerator
{
public:
	BarrierGenerator(const HardwareProfile& profile) : m_profile(profile) {}

	SASS::Instruction *Generate(SASS::Schedule::Barrier barrier, std::uint8_t scoreboard = 0, std::uint8_t latency = 0) const;

private:
	template<class T>
	T *Generate(SASS::Schedule::Barrier barrier, std::uint8_t scoreboard, std::uint8_t latency) const;

	template<class T>
	typename T::Barrier GetInstructionBarrier(SASS::Schedule::Barrier barrier) const;

	const HardwareProfile& m_profile;
};

}
}
