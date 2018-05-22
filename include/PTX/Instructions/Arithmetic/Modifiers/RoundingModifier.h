#pragma once

namespace PTX
{

template<class T, bool R = false>
class RoundingModifier
{
public:
	RoundingModifier() {}
	RoundingModifier(typename T::RoundingMode roundingMode) : m_roundingMode(roundingMode) {}

	typename T::RoundingMode GetRoundingMode() const { return m_roundingMode; }
	void SetRoundingMode(typename T::RoundingMode roundingMode) { m_roundingMode = roundingMode; }

protected:
	typename T::RoundingMode m_roundingMode = T::RoundingMode::None;
};

template<class T>
class RoundingModifier<T, true>
{
public:
	RoundingModifier(typename T::RoundingMode roundingMode) : m_roundingMode(roundingMode)
	{
		if (roundingMode == T::RoundingMode::None)
		{
			std::cerr << "PTX::RoundingModifier requires rounding mode" << std::endl;
			std::exit(EXIT_FAILURE);
		}
	}

	typename T::RoundingMode GetRoundingMode() const { return m_roundingMode; }
	void SetRoundingMode(typename T::RoundingMode roundingMode)
	{
		if (roundingMode == T::RoundingMode::None)
		{
			std::cerr << "PTX::RoundingModifier requires rounding mode" << std::endl;
			std::exit(EXIT_FAILURE);
		}
	       	m_roundingMode = roundingMode;
	}

protected:
	typename T::RoundingMode m_roundingMode = T::RoundingMode::None;
};

}
