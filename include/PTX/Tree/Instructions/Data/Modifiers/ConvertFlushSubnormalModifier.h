#pragma once

namespace PTX {

template<class D, class S, typename Enable = void>
class ConvertFlushSubnormalModifier
{
public:
	constexpr static bool Enabled = false;
};

template<class D, class S>
class ConvertFlushSubnormalModifier<D, S, std::enable_if_t<(std::is_same<D, Float32Type>::value || std::is_same<S, Float32Type>::value)>>
{
public:
	constexpr static bool Enabled = true;

	ConvertFlushSubnormalModifier(bool flush = false) : m_flush(flush) {}

	bool GetFlushSubnormal() const { return m_flush; }
	void SetFlushSubnormal(bool flush) { m_flush = flush; }

	std::string OpCodeModifier() const
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
