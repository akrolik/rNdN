#include "CUDA/Event.h"

#include "CUDA/Utils.h"

namespace CUDA {

Event::Event()
{
	checkRuntimeError(cudaEventCreate(&m_event));
}

Event::~Event()
{
	checkRuntimeError(cudaEventDestroy(m_event));
}

void Event::Record()
{
	checkRuntimeError(cudaEventRecord(m_event, 0));
}

void Event::Synchronize()
{
	checkRuntimeError(cudaEventSynchronize(m_event));
}

long long Event::Time(const Event& start, const Event& end)
{
	float time_ms;
	checkRuntimeError(cudaEventElapsedTime(&time_ms, start.m_event, end.m_event));
	return static_cast<long long>(time_ms * 1000 * 1000);
}

}
