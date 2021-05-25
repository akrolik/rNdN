#include "Backend/Scheduler/BlockScheduler.h"

namespace Backend {
namespace Scheduler {

void BlockScheduler::Schedule(SASS::Function *function)
{
	m_function = function;
	for (auto& block : function->GetBasicBlocks())
	{
		ScheduleBlock(block);
	}
	function = nullptr;
}

}
}
