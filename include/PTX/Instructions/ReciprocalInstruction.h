#pragma once

#include "PTX/Instructions/PredicatedInstruction.h"

#include "PTX/Operands/Operand.h"
#include "PTX/Operands/Variables/Register.h"

namespace PTX {

template<class T>
class ReciprocalInstruction : public PredicatedInstruction
{
	DISABLE_ALL(ReciprocalInstruction, T);
};

template<>
class ReciprocalInstruction<Float32Type> : public PredicatedInstruction
{
public:
	ReciprocalInstruction(Register<Float32Type> *destination, Operand<Float32Type> *source, Float32Type::RoundingMode roundingMode = Float32Type::RoundingMode::None) : m_destination(destination), m_source(source), m_roundingMode(roundingMode)
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
				return "rcp.approx.ftz" + Float32Type::Name();
			}
			return "rcp.approx" + Float32Type::Name();
		}
		else
		{
			if (m_flush)
			{
				return "rcp" + Float32Type::RoundingModeString(m_roundingMode) + ".ftz" + Float32Type::Name();
			}
			return "rcp" + Float32Type::RoundingModeString(m_roundingMode) + Float32Type::Name();
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
class ReciprocalInstruction<Float64Type> : public PredicatedInstruction
{
public:
	ReciprocalInstruction(Register<Float64Type> *destination, Operand<Float64Type> *source, Float64Type::RoundingMode roundingMode = Float64Type::RoundingMode::None) : m_destination(destination), m_source(source), m_roundingMode(roundingMode)
	{
		if (m_roundingMode == Float64Type::RoundingMode::None)
		{
			m_approximate = true;
		}
	}

	void SetApproximate(bool approximate)
	{
		m_approximate = approximate;
		m_roundingMode = Float64Type::RoundingMode::None;
	}

	void SetRoundingMode(Float64Type::RoundingMode roundingMode)
	{
		if (roundingMode == Float64Type::RoundingMode::None)
		{
			m_approximate = true;
		}
		else
		{
			m_approximate = false;
		}
		m_roundingMode = roundingMode;
	}

	std::string OpCode() const
	{
		if (m_approximate)
		{
			return "rcp.approx.ftz" + Float64Type::Name();
		}
		return "rcp" + Float64Type::RoundingModeString(m_roundingMode) + Float64Type::Name();
	}

	std::string Operands() const
	{
		return m_destination->ToString() + ", " + m_source->ToString();
	}
private:
	Register<Float64Type> *m_destination = nullptr;
	Operand<Float64Type> *m_source = nullptr;

	bool m_approximate = false;
	Float64Type::RoundingMode m_roundingMode = Float64Type::RoundingMode::None;
};

}
