#pragma once

#include "Backend/Scheduler/BlockScheduler.h"

namespace Backend {
namespace Scheduler {

class LinearBlockScheduler : public BlockScheduler
{
protected:
	void ScheduleBlock(SASS::BasicBlock *block) override;
};

}
}
