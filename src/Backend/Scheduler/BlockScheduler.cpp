#include "Backend/Scheduler/BlockScheduler.h"

namespace Backend {
namespace Scheduler {

void BlockScheduler::Schedule(SASS::Function *function)
{
	for (auto& block : function->GetBasicBlocks())
	{
		ScheduleBlock(block);
	}
}

}
}
