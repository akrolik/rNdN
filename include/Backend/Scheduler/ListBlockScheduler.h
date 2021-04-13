#pragma once

#include "Backend/Scheduler/BlockScheduler.h"

namespace Backend {
namespace Scheduler {

class ListBlockScheduler : public BlockScheduler
{
protected:
	void ScheduleBlock(SASS::BasicBlock *block) override;
};

}
}
