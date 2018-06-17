#pragma once

namespace PTX {

template<class T, bool force = false, typename Enable = void>
class SaturateModifier
{
};

template<class T, bool force>
class SaturateModifier<T, force, std::enable_if_t<force || T::SaturateModifier>>
{
public:
	SaturateModifier() {}
	SaturateModifier(bool saturate) : m_saturate(saturate) {}

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
