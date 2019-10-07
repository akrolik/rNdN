#pragma once

#include <cuda.h>
#include <cuda_runtime.h>

namespace CUDA {

class Chrono;
class Event
{
	friend class Chrono;
public:
	Event();
	~Event();

	void Record();
	void Synchronize();

private:
	cudaEvent_t m_event;
};

}
