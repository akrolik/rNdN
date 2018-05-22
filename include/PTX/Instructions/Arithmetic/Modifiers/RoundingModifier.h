#pragma once

template<class T>
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
