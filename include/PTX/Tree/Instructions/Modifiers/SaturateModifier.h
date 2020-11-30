#pragma once

namespace PTX {

template<class T, bool force = false, typename Enable = void>
class SaturateModifier
{
public:
	constexpr static bool Enabled = false;
};

template<class T, bool force>
class SaturateModifier<T, force,
      std::enable_if_t<REQUIRE_EXACT(T, Int32Type, Float16Type, Float16x2Type, Float32Type) || force>>
{
public:
	constexpr static bool Enabled = true;

	SaturateModifier(bool saturate = false) : m_saturate(saturate) {}

	bool GetSaturate() const { return m_saturate; }
	void SetSaturate(bool saturate) { m_saturate = saturate; }

	std::string OpCodeModifier() const
	{
		if (m_saturate)
		{
			return ".sat";
		}
		return "";
	}

protected:
	bool m_saturate = false;
};

}
