#include "Utils/Chrono.h"

#include "Utils/Logger.h"

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

	Utils::Logger::LogSection("Execution time");
	Utils::Logger::LogTiming(overall);
}

const Chrono::SpanTiming *Chrono::Start(const std::string& name)
{
	auto& instance = GetInstance();

	auto timing = new Chrono::SpanTiming(name);
	instance.m_timings.top()->AddChild(timing);
	instance.m_timings.push(timing);

	timing->Start();
	return timing;
}

void Chrono::End(const SpanTiming *start)
{
	auto& instance = GetInstance();
	auto stackStart = instance.m_timings.top();
	if (stackStart != start)
	{
		Utils::Logger::LogError("Received end time '" + start->GetName() + "' but expected '" + stackStart->GetName() + "'");
	}
	stackStart->End();
	instance.m_timings.pop();
}

void Chrono::AddPointTiming(const std::string& name, long time)
{
	auto& instance = GetInstance();
	auto timing = new Chrono::PointTiming(name, time);
	instance.m_timings.top()->AddChild(timing);
}

void Chrono::SpanTiming::Start()
{
	m_start = std::chrono::steady_clock::now();
}

void Chrono::SpanTiming::End()
{
	m_end = std::chrono::steady_clock::now();
}

long Chrono::SpanTiming::GetTime() const
{
	return std::chrono::duration_cast<std::chrono::microseconds>(m_end - m_start).count();
}

}
