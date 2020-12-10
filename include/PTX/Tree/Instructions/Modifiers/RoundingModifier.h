#pragma once

#include "PTX/Tree/Type.h"

namespace PTX {

template<class T, bool R = false, typename Enable = void>
class RoundingModifier
{
public:
	constexpr static bool Enabled = false;
};

template<class T>
class RoundingModifier<T, false, std::enable_if_t<is_rounding_type<T>::value>>
{
public:
	constexpr static bool Enabled = true;

	RoundingModifier() {}
	RoundingModifier(typename T::RoundingMode roundingMode) : m_roundingMode(roundingMode) {}

	// Properties

	typename T::RoundingMode GetRoundingMode() const { return m_roundingMode; }
	void SetRoundingMode(typename T::RoundingMode roundingMode) { m_roundingMode = roundingMode; }

	// Formatting

	std::string GetOpCodeModifier() const
	{
		return T::RoundingModeString(m_roundingMode);
	}

	bool IsActive() const
	{
		return m_roundingMode != T::RoundingMode::None;
	}

protected:
	typename T::RoundingMode m_roundingMode = T::RoundingMode::None;
};

template<class T>
class RoundingModifier<T, true, std::enable_if_t<is_rounding_type<T>::value>>
{
public:
	constexpr static bool Enabled = true;

	RoundingModifier(typename T::RoundingMode roundingMode) : m_roundingMode(roundingMode)
	{
		if (roundingMode == T::RoundingMode::None)
		{
			Utils::Logger::LogError("PTX::RoundingModifier requires rounding mode");
		}
	}

	// Properties

	typename T::RoundingMode GetRoundingMode() const { return m_roundingMode; }
	void SetRoundingMode(typename T::RoundingMode roundingMode)
	{
		if (roundingMode == T::RoundingMode::None)
		{
			Utils::Logger::LogError("PTX::RoundingModifier requires rounding mode");
		}
	       	m_roundingMode = roundingMode;
	}

	// Formatting

	std::string GetOpCodeModifier() const
	{
		return T::RoundingModeString(m_roundingMode);
	}

	bool IsActive() const
	{
		return m_roundingMode != T::RoundingMode::None;
	}

protected:
	typename T::RoundingMode m_roundingMode = T::RoundingMode::None;
};

}
