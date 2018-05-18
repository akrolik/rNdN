#pragma once

#include "PTX/Instructions/PredicatedInstruction.h"

#include "PTX/Operands/Operand.h"
#include "PTX/Operands/Variables/Register.h"

namespace PTX {

template<class T>
class FMAInstruction : public PredicatedInstruction
{
	REQUIRE_TYPES(FMAInstruction, FloatType);
public:
	FMAInstruction(Register<T> *destination, Operand<T> *sourceA, Operand<T> *sourceB, Operand<T> *sourceC, typename T::RoundingMode roundingMode) : m_destination(destination), m_sourceA(sourceA), m_sourceB(sourceB), m_sourceC(sourceC), m_roundingMode(roundingMode)
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

	std::string Operands() const
	{
		return m_destination->ToString() + ", " + m_sourceA->ToString() + ", " + m_sourceB->ToString() + ", " + m_sourceC->ToString();
	}

private:
	Register<T> *m_destination = nullptr;
	Operand<T> *m_sourceA = nullptr;
	Operand<T> *m_sourceB = nullptr;
	Operand<T> *m_sourceC = nullptr;

	typename T::RoundingMode m_roundingMode = T::RoundingMode::None;
	bool m_flush = false;
	bool m_saturate = false;
};

template<>
class FMAInstruction<Float64Type> : public PredicatedInstruction
{
public:
	FMAInstruction(Register<Float64Type> *destination, Operand<Float64Type> *sourceA, Operand<Float64Type> *sourceB, Operand<Float64Type> *sourceC, Float64Type::RoundingMode roundingMode) : m_destination(destination), m_sourceA(sourceA), m_sourceB(sourceB), m_sourceC(sourceC), m_roundingMode(roundingMode)
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

	std::string Operands() const
	{
		return m_destination->ToString() + ", " + m_sourceA->ToString() + ", " + m_sourceB->ToString() + ", " + m_sourceC->ToString();
	}

private:
	Register<Float64Type> *m_destination = nullptr;
	Operand<Float64Type> *m_sourceA = nullptr;
	Operand<Float64Type> *m_sourceB = nullptr;
	Operand<Float64Type> *m_sourceC = nullptr;

	Float64Type::RoundingMode m_roundingMode = Float64Type::RoundingMode::None;
};

}
