#pragma once

#include "SASS/SASS.h"

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
