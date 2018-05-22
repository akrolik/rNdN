#pragma once

namespace PTX {

class HalfModifier
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

protected:
	bool m_upper = false;
	bool m_lower = false;
};

}
