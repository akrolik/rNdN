#pragma once

#include <cuda.h>
#include <cuda_runtime.h>

namespace CUDA {

class Event
{
public:
	Event();
	~Event();

	void Record();
	void Synchronize();

	static long long Time(const Event& start, const Event& end);

private:
	cudaEvent_t m_event;
};

}
