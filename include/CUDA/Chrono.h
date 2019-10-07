#pragma once

#include "CUDA/Event.h"

namespace CUDA {

class Chrono
{
public:
	static Event *Start();
	static long End(Event *start);

private:
	Chrono() {}
};

}
