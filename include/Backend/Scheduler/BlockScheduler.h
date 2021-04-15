#pragma once

#include "SASS/Tree/Tree.h"

namespace Backend {
namespace Scheduler {

class BlockScheduler
{         
public:
	void Schedule(SASS::Function *function);

protected:
	virtual void ScheduleBlock(SASS::BasicBlock *block) = 0;
};

}
}
