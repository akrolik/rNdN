#pragma once

#include <iostream>

namespace PTX
{

template<class D, class S, typename Enable = void>
class ConvertSaturateModifier
{
public:
	constexpr static bool Enabled = false;
};

template<class D, class S>
class ConvertSaturateModifier<D, S, std::enable_if_t<
	(is_int_type<D>::value && is_int_type<S>::value && D::BitSize < S::BitSize) || is_float_type<D>::value>>
{
public:
	constexpr static bool Enabled = true;

	ConvertSaturateModifier(bool saturate = false) : m_saturate(saturate) {}

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
