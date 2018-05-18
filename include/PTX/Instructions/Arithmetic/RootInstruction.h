#pragma once

#include "PTX/Instructions/PredicatedInstruction.h"

#include "PTX/Operands/Operand.h"
#include "PTX/Operands/Variables/Register.h"

namespace PTX {

template<class T>
class RootInstruction : public PredicatedInstruction
{
	DISABLE_ALL(RootInstruction, T);
};

template<>
class RootInstruction<Float32Type> : public PredicatedInstruction
{
public:
	RootInstruction(Register<Float32Type> *destination, Operand<Float32Type> *source, Float32Type::RoundingMode roundingMode = Float32Type::RoundingMode::None) : m_destination(destination), m_source(source), m_roundingMode(roundingMode)
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

	std::string Operands() const
	{
		return m_destination->ToString() + ", " + m_source->ToString();
	}
private:
	Register<Float32Type> *m_destination = nullptr;
	Operand<Float32Type> *m_source = nullptr;

	bool m_approximate = false;
	Float32Type::RoundingMode m_roundingMode = Float32Type::RoundingMode::None;
	bool m_flush = false;
};

template<>
class RootInstruction<Float64Type> : public PredicatedInstruction
{
public:
	RootInstruction(Register<Float64Type> *destination, Operand<Float64Type> *source, Float64Type::RoundingMode roundingMode) : m_destination(destination), m_source(source), m_roundingMode(roundingMode)
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

	std::string Operands() const
	{
		return m_destination->ToString() + ", " + m_source->ToString();
	}
private:
	Register<Float64Type> *m_destination = nullptr;
	Operand<Float64Type> *m_source = nullptr;

	Float64Type::RoundingMode m_roundingMode = Float64Type::RoundingMode::None;
};

}
