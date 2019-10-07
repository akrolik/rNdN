#include "CUDA/Chrono.h"

#include "CUDA/Utils.h"

namespace CUDA {

Event *Chrono::Start()
{
	auto event = new Event();
	event->Record();
	return event;
}

long Chrono::End(Event *start)
{
	Event end;
	end.Record();
	end.Synchronize();

	float time_ms;
	checkRuntimeError(cudaEventElapsedTime(&time_ms, start->m_event, end.m_event));
	delete start;

	return long(time_ms * 1000);
}

}


