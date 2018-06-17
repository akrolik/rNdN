#pragma once

namespace PTX {

template<class T, bool force = false, typename Enable = void>
class HalfModifier
{
};

template<class T, bool force>
class HalfModifier<T, force, std::enable_if_t<force || T::HalfModifier>>
{
public:
	HalfModifier() {}

	bool GetLower() const { return m_lower; }
	void SetLower(bool lower)
	{
		m_lower = lower;
		if (lower)
		{
			m_upper = false;
		}
	}

	bool GetUpper() const { return m_upper; }
	void SetUpper(bool upper)
	{
		m_upper = upper;
		if (upper)
		{
			m_lower = false;
		}
	}

	std::string OpCodeModifier() const
	{
		if (m_upper)
		{
			return ".hi";
		}
		else if (m_lower)
		{
			return ".lo";
		}
		return "";
	}

protected:
	bool m_upper = false;
	bool m_lower = false;
};

}
