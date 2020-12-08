#pragma once

namespace PTX {

template<class D, class S, typename Enable = void>
class ConvertRoundingModifier
{
public:
	constexpr static bool Enabled = false;
};

template<class D, class S>
class ConvertRoundingModifier<D, S, std::enable_if_t<
	(is_int_type<D>::value && is_float_type<S>::value) ||
	(is_float_type<D>::value && std::is_same<D, S>::value)>>
{
public:
	constexpr static bool Enabled = true;

	enum class RoundingMode {
		None,
		Nearest,
		Zero,
		NegativeInfinity,
		PositiveInfinity
	};

	static std::string RoundingModeString(RoundingMode roundingMode)
	{
		switch (roundingMode)
		{
			case RoundingMode::None:
				return "";
			case RoundingMode::Nearest:
				return ".rni";
			case RoundingMode::Zero:
				return ".rzi";
			case RoundingMode::NegativeInfinity:
				return ".rmi";
			case RoundingMode::PositiveInfinity:
				return ".rpi";
		}
		return ".<unknown>";
	}

	ConvertRoundingModifier() {}
	ConvertRoundingModifier(RoundingMode roundingMode) : m_roundingMode(roundingMode) {}

	RoundingMode GetRoundingMode() const { return m_roundingMode; }
	void SetRoundingMode(RoundingMode roundingMode) { m_roundingMode = roundingMode; }

	std::string OpCodeModifier() const
	{
		return RoundingModeString(m_roundingMode);
	}

	bool IsActive() const
	{
		return m_roundingMode != RoundingMode::None;
	}

protected:
	RoundingMode m_roundingMode = RoundingMode::None;
};

template<class D, class S>
class ConvertRoundingModifier<D, S, std::enable_if_t<
	(is_float_type<D>::value && is_int_type<S>::value) ||
	(is_float_type<D>::value && BitSize<D::TypeBits>::NumBits < BitSize<S::TypeBits>::NumBits)>>
{
public:
	constexpr static bool Enabled = true;

	enum class RoundingMode {
		None,
		Nearest,
		Zero,
		NegativeInfinity,
		PositiveInfinity
	};

	static std::string RoundingModeString(RoundingMode roundingMode)
	{
		switch (roundingMode)
		{
			case RoundingMode::None:
				return "";
			case RoundingMode::Nearest:
				return ".rn";
			case RoundingMode::Zero:
				return ".rz";
			case RoundingMode::NegativeInfinity:
				return ".rm";
			case RoundingMode::PositiveInfinity:
				return ".rp";
		}
		return ".<unknown>";
	}

	ConvertRoundingModifier() {}
	ConvertRoundingModifier(RoundingMode roundingMode) : m_roundingMode(roundingMode) {}

	RoundingMode GetRoundingMode() const { return m_roundingMode; }
	void SetRoundingMode(RoundingMode roundingMode) { m_roundingMode = roundingMode; }

	std::string OpCodeModifier() const
	{
		return RoundingModeString(m_roundingMode);
	}

	bool IsActive() const
	{
		return m_roundingMode != RoundingMode::None;
	}

protected:
	RoundingMode m_roundingMode = RoundingMode::None;
};

}
