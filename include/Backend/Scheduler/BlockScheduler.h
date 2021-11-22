#pragma once

#include "Backend/Scheduler/BarrierGenerator.h"
#include "Backend/Scheduler/Profiles/HardwareProfile.h"

#include "SASS/Tree/Tree.h"

namespace Backend {
namespace Scheduler {

class BlockScheduler
{         
public:
	BlockScheduler(const HardwareProfile& profile) : m_profile(profile), m_barrierGenerator(m_profile) {}

	void Schedule(SASS::Function *function);

protected:
	virtual void ScheduleBlock(SASS::BasicBlock *block) = 0;
	SASS::Function *m_function = nullptr;

	const HardwareProfile& m_profile;
	BarrierGenerator m_barrierGenerator;
};

}
}
