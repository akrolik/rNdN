#pragma once

namespace PTX {

template<class T, bool force = false, typename Enable = void>
class HalfModifier
{
public:
	constexpr static bool Enabled = false;
};

template<class T, bool force>
class HalfModifier<T, force,
      std::enable_if_t<REQUIRE_EXACT(T, Int8Type, Int16Type, Int32Type, Int64Type, UInt8Type, UInt16Type, UInt64Type) || force>>
{
public:
	constexpr static bool Enabled = true;

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

	bool IsActive() const
	{
		return m_lower || m_upper;
	}

protected:
	bool m_upper = false;
	bool m_lower = false;
};

}
