#include "Utils/Chrono.h"

#include "Utils/Logger.h"
#include "Utils/Options.h"

namespace Utils {

void Chrono::Initialize()
{
	auto& instance = GetInstance();
	auto timing = new Chrono::SpanTiming("Execution time");
	instance.m_timings.push(timing);
	timing->Start();
}

void Chrono::Complete()
{
	auto& instance = GetInstance();
	auto overall = instance.m_timings.top();
	overall->End();
	instance.m_timings.pop();

	if (Utils::Options::Present(Utils::Options::Opt_Print_time))
	{
		Utils::Logger::LogSection("Execution time");
		Utils::Logger::LogTiming(overall);
	}
}

void Chrono::Pause(const SpanTiming *timing)
{
	auto& instance = GetInstance();
	auto stackStart = instance.m_timings.top();
	if (stackStart != timing)
	{
		Error(timing, stackStart);
	}

	stackStart->End();
	instance.m_timings.pop();
	instance.m_pausedTimings.insert(static_cast<SpanTiming *>(stackStart));
}

void Chrono::Continue(const SpanTiming *timing)
{
	auto& instance = GetInstance();
	auto search = instance.m_pausedTimings.find(const_cast<SpanTiming *>(timing));
	if (search == instance.m_pausedTimings.end())
	{
		Utils::Logger::LogError("Timing '" + timing->GetName() + "' not paused");
	}
	auto pausedTiming = (*search);

	auto stackStart = instance.m_timings.top();
	if (stackStart != timing->GetParent())
	{
		Error(timing->GetParent(), stackStart);
	}

	pausedTiming->Start();
	instance.m_pausedTimings.erase(search);
	instance.m_timings.push(pausedTiming);
}

void Chrono::SpanTiming::Start()
{
	m_start = std::chrono::steady_clock::now();
}

void Chrono::SpanTiming::End()
{
	auto end = std::chrono::steady_clock::now();
	m_elapsed += std::chrono::duration_cast<std::chrono::nanoseconds>(end - m_start).count();
}

long long Chrono::SpanTiming::GetTime() const
{
	return m_elapsed;
}

void Chrono::CUDATiming::Start()
{
	m_start.Record();
}

void Chrono::CUDATiming::End()
{
	m_end.Record();
}

long long Chrono::CUDATiming::GetTime() const
{
	m_start.Synchronize();
	m_end.Synchronize();
	return CUDA::Event::Time(m_start, m_end);
}

void Chrono::Error(const Timing *start, const Timing *stack)
{
	Utils::Logger::LogError("Received end time '" + start->GetName() + "' but expected '" + stack->GetName() + "'");
}

}
