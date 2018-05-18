#pragma once

#include "PTX/Instructions/InstructionBase.h"

namespace PTX {

template<class T>
class FMAInstruction : public InstructionBase<T, 3>
{
	REQUIRE_TYPES(FMAInstruction, FloatType);
public:
	FMAInstruction(Register<T> *destination, Operand<T> *sourceA, Operand<T> *sourceB, Operand<T> *sourceC, typename T::RoundingMode roundingMode) : InstructionBase<T, 3>(destination, sourceA, sourceB, sourceC), m_roundingMode(roundingMode)
	{
		if (m_roundingMode == T::RoundingMode::None)
		{
			std::cerr << "PTX::FMAInstruction requires rounding mode " << std::endl;
			std::exit(EXIT_FAILURE);
		}
	}

	void SetRoundingMode(typename T::RoundingMode roundingMode)
	{
		if (m_roundingMode == T::RoundingMode::None)
		{
			std::cerr << "PTX::FMAInstruction requires rounding mode " << std::endl;
			std::exit(EXIT_FAILURE);
		}

	}

	void SetFlush(bool flush) { m_flush = flush; }

	void SetSaturate(bool saturate) { m_saturate = saturate; }

	std::string OpCode() const
	{
		return "fma" + T::RoundingModeString(m_roundingMode) + ((m_flush) ? ".ftz" : "") + ((m_saturate) ? ".sat" : "") + T::Name();
	}

private:
	typename T::RoundingMode m_roundingMode = T::RoundingMode::None;
	bool m_flush = false;
	bool m_saturate = false;
};

template<>
class FMAInstruction<Float64Type> : public InstructionBase<Float64Type, 3>
{
public:
	FMAInstruction(Register<Float64Type> *destination, Operand<Float64Type> *sourceA, Operand<Float64Type> *sourceB, Operand<Float64Type> *sourceC, Float64Type::RoundingMode roundingMode) : InstructionBase<Float64Type, 3>(destination, sourceA, sourceB, sourceC), m_roundingMode(roundingMode)
	{
		if (m_roundingMode == Float64Type::RoundingMode::None)
		{
			std::cerr << "PTX::FMAInstruction requires rounding mode" << std::endl;
			std::exit(EXIT_FAILURE);
		}
	}

	void SetRoundingMode(Float64Type::RoundingMode roundingMode)
	{
		if (m_roundingMode == Float64Type::RoundingMode::None)
		{
			std::cerr << "PTX::FMAInstruction requires rounding mode" << std::endl;
			std::exit(EXIT_FAILURE);
		}
	}

	std::string OpCode() const
	{
		return "fma" + Float64Type::RoundingModeString(m_roundingMode) + Float64Type::Name();
	}

private:
	Float64Type::RoundingMode m_roundingMode = Float64Type::RoundingMode::None;
};

}
