#pragma once

#include "PTX/Instructions/InstructionBase.h"

namespace PTX {

template<class T>
class RootInstruction : public InstructionBase<T, 1>
{
	DISABLE_ALL(RootInstruction, T);
};

template<>
class RootInstruction<Float32Type> : public InstructionBase<Float32Type, 1>
{
public:
	RootInstruction(Register<Float32Type> *destination, Operand<Float32Type> *source, Float32Type::RoundingMode roundingMode = Float32Type::RoundingMode::None) : InstructionBase<Float32Type, 1>(destination, source), m_roundingMode(roundingMode)
	{
		if (m_roundingMode == Float32Type::RoundingMode::None)
		{
			m_approximate = true;
		}
	}

	void SetApproximate(bool approximate)
	{
		m_approximate = approximate;
		m_roundingMode = Float32Type::RoundingMode::None;
	}

	void SetRoundingMode(Float32Type::RoundingMode roundingMode)
	{
		if (roundingMode == Float32Type::RoundingMode::None)
		{
			m_approximate = true;
		}
		else
		{
			m_approximate = false;
		}
		m_roundingMode = roundingMode;
	}

	void SetFlushSubNormal(bool flush) { m_flush = flush; }

	std::string OpCode() const
	{
		if (m_approximate)
		{
			if (m_flush)
			{
				return "sqrt.approx.ftz" + Float32Type::Name();
			}
			return "sqrt.approx" + Float32Type::Name();
		}
		else
		{
			if (m_flush)
			{
				return "sqrt" + Float32Type::RoundingModeString(m_roundingMode) + ".ftz" + Float32Type::Name();
			}
			return "sqrt" + Float32Type::RoundingModeString(m_roundingMode) + Float32Type::Name();
		}
	}

private:
	bool m_approximate = false;
	Float32Type::RoundingMode m_roundingMode = Float32Type::RoundingMode::None;
	bool m_flush = false;
};

template<>
class RootInstruction<Float64Type> : public InstructionBase<Float64Type, 1>
{
public:
	RootInstruction(Register<Float64Type> *destination, Operand<Float64Type> *source, Float64Type::RoundingMode roundingMode) : InstructionBase<Float64Type, 1>(destination, source), m_roundingMode(roundingMode)
	{
		if (m_roundingMode == Float64Type::RoundingMode::None)
		{
			std::cerr << "PTX::RootInstruction requires rounding mode with Float64Type" << std::endl;
			std::exit(EXIT_FAILURE);
		}
	}

	void SetRoundingMode(Float64Type::RoundingMode roundingMode)
	{
		if (m_roundingMode == Float64Type::RoundingMode::None)
		{
			std::cerr << "PTX::RootInstruction requires rounding mode with Float64Type" << std::endl;
			std::exit(EXIT_FAILURE);
		}
		m_roundingMode = roundingMode;
	}

	std::string OpCode() const
	{
		return "rcp" + Float64Type::RoundingModeString(m_roundingMode) + Float64Type::Name();
	}

private:
	Float64Type::RoundingMode m_roundingMode = Float64Type::RoundingMode::None;
};

}
