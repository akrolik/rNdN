#pragma once

#include <chrono>
#include <stack>
#include <string>
#include <vector>

namespace Utils {

class Chrono
{
public:
	using TimeTy = std::chrono::time_point<std::chrono::steady_clock>;

	class Timing
	{
	public:
		Timing(const std::string& name) : m_name(name) {}

		const std::string& GetName() const { return m_name; }

		bool HasChildren() const { return (m_children.size() > 0); }
		const std::vector<Timing *>& GetChildren() const { return m_children; }
		void AddChild(Timing *timing) { m_children.push_back(timing); }

		virtual long GetTime() const = 0;

	protected:
		std::vector<Timing *> m_children;
		std::string m_name;
	};

	class PointTiming : public Timing
	{
	public:
		PointTiming(const std::string& name, long time) : Timing(name), m_time(time) {}

		long GetTime() const override { return m_time; }

	protected:
		long m_time = 0;
	};

	class SpanTiming : public Timing
	{
	public:
		SpanTiming(const std::string& name) : Timing(name) {}

		void Start();
		void End();

		long GetTime() const override;

	protected:
		TimeTy m_start;
		TimeTy m_end;
	};

	static void Initialize();
	static void Complete();

	static const SpanTiming *Start(const std::string& name);
	static void End(const SpanTiming *start);

	static void AddPointTiming(const std::string& name, long time);

private:
	Chrono() {}

	static Chrono& GetInstance()
	{
		static Chrono instance;
		return instance;
	}

	static void PrintTiming(const Timing *timing);

	std::stack<SpanTiming *> m_timings;
};

class ScopedChrono
{
public:
	ScopedChrono(const std::string& name)
	{
		m_timing = Chrono::Start(name);
	}

	~ScopedChrono()
	{
		Chrono::End(m_timing);
	}

private:
	const Chrono::SpanTiming *m_timing = nullptr;
};

}
