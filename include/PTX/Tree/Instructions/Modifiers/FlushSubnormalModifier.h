#pragma once

namespace PTX {

template<class T, bool force = false, typename Enable = void>
class FlushSubnormalModifier
{
public:
	constexpr static bool Enabled = false;
};

template<class T, bool force>
class FlushSubnormalModifier<T, force,
      std::enable_if_t<REQUIRE_EXACT(T, Float16Type, Float16x2Type, Float32Type) || force>>
{
public:
	constexpr static bool Enabled = true;

	FlushSubnormalModifier(bool flush = false) : m_flush(flush) {}

	// Properties

	bool GetFlushSubnormal() const { return m_flush; }
	void SetFlushSubnormal(bool flush) { m_flush = flush; }

	// Formatting

	std::string GetOpCodeModifier() const
	{
		if (m_flush)
		{
			return ".ftz";
		}
		return "";
	}

protected:
	bool m_flush = false;
};

}
