#pragma once

#include <chrono>
#include <stack>
#include <string>
#include <unordered_set>
#include <vector>

#include "CUDA/Event.h"

namespace Utils {

class Chrono
{
public:
	using TimeTy = std::chrono::time_point<std::chrono::steady_clock>;

	class Timing
	{
	public:
		Timing(const std::string& name, const Timing *parent = nullptr) : m_name(name), m_parent(parent) {}

		const std::string& GetName() const { return m_name; }

		bool HasChildren() const { return (m_children.size() > 0); }
		void AddChild(Timing *timing) { m_children.push_back(timing); }

		std::vector<const Timing *> GetChildren() const
		{
			return { std::begin(m_children), std::end(m_children) };
		}
		std::vector<Timing *>& GetChildren() { return m_children; }

		const Timing *GetParent() const { return m_parent; }

		virtual void Start() = 0;
		virtual void End() = 0;

		virtual long long GetTime() const = 0;

	protected:
		const Timing *m_parent = nullptr;
		std::vector<Timing *> m_children;

		std::string m_name;
	};

	class SpanTiming : public Timing
	{
	public:
		using Timing::Timing;

		void Start() override;
		void End() override;

		long long GetTime() const override;

	protected:
		TimeTy m_start;
		long long m_elapsed = 0;
	};

	class CUDATiming : public Timing
	{
	public:
		using Timing::Timing;

		void Start() override;
		void End() override;

		long long GetTime() const override;

	protected:
		mutable CUDA::Event m_start;
		mutable CUDA::Event m_end;
	};

	static void Initialize();
	static void Complete();

	template<class T = SpanTiming>
	static const T *Start(const std::string& name)
	{
		auto& instance = GetInstance();

		auto parent = instance.m_timings.top();
		auto timing = new T(name, parent);
		parent->AddChild(timing);
		instance.m_timings.push(timing);

		timing->Start();
		return timing;
	}
	static const CUDATiming *StartCUDA(const std::string& name) { return Start<CUDATiming>(name); }

	template<class T = SpanTiming>
	static void End(const T *start)
	{
		auto& instance = GetInstance();
		auto stackStart = instance.m_timings.top();
		if (stackStart != start)
		{
			Error(start, stackStart);
		}
		stackStart->End();
		instance.m_timings.pop();
	}

	static void Pause(const SpanTiming *timing);
	static void Continue(const SpanTiming *timing);

private:
	Chrono() {}

	static Chrono& GetInstance()
	{
		static Chrono instance;
		return instance;
	}

	static void Error(const Timing *start, const Timing *stack);
	static void PrintTiming(const Timing *timing);

	std::stack<Timing *> m_timings;
	std::unordered_set<SpanTiming *> m_pausedTimings;
};

template<class T = Chrono::SpanTiming>
class ScopedChrono
{
public:
	ScopedChrono(const std::string& name)
	{
		m_timing = Chrono::Start<T>(name);
	}

	~ScopedChrono()
	{
		Chrono::End<T>(m_timing);
	}

private:
	const T *m_timing = nullptr;
};

}
