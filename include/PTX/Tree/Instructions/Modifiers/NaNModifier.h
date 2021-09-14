#pragma once

namespace PTX {

template<class T, bool force = false, typename Enable = void>
class NaNModifier
{
public:
	constexpr static bool Enabled = false;
};

template<class T, bool force>
class NaNModifier<T, force,
      std::enable_if_t<REQUIRE_EXACT(T, Float16Type, Float16x2Type, Float32Type) || force>>
{
public:
	constexpr static bool Enabled = true;

	NaNModifier(bool nan = false) : m_nan(nan) {}

	// Properties

	bool GetNaN() const { return m_nan; }
	void SetNaN(bool nan) { m_nan = nan; }

	// Formatting

	std::string GetOpCodeModifier() const
	{
		if (m_nan)
		{
			return ".NaN";
		}
		return "";
	}

protected:
	bool m_nan = false;
};

}
