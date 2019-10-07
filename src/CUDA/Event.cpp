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

}
