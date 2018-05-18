#pragma once

#include "PTX/Instructions/PredicatedInstruction.h"

#include "PTX/Operands/Operand.h"
#include "PTX/Operands/Variables/Register.h"

namespace PTX {

template<class T>
class DivideInstruction : public PredicatedInstruction
{
	REQUIRE_TYPE(DivideInstruction, ScalarType);
	DISABLE_TYPE(DivideInstruction, Int8Type);
	DISABLE_TYPE(DivideInstruction, UInt8Type);
	DISABLE_TYPE(DivideInstruction, Float16Type);
public:
	DivideInstruction(Register<T> *destination, Operand<T> *sourceA, Operand<T> *sourceB) : m_destination(destination), m_sourceA(sourceA), m_sourceB(sourceB) {}

	std::string OpCode() const
	{
		return "div" + T::Name();
	}

	std::string Operands() const
	{
		return m_destination->ToString() + ", " + m_sourceA->ToString() + ", " + m_sourceB->ToString();
	}

private:
	Register<T> *m_destination = nullptr;
	Operand<T> *m_sourceA = nullptr;
	Operand<T> *m_sourceB = nullptr;
};

template<>
class DivideInstruction<Float32Type> : public PredicatedInstruction
{
public:
	DivideInstruction(Register<Float32Type> *destination, Operand<Float32Type> *sourceA, Operand<Float32Type> *sourceB, Float32Type::RoundingMode roundingMode = Float32Type::RoundingMode::None) : m_destination(destination), m_sourceA(sourceA), m_sourceB(sourceB), m_roundingMode(roundingMode) {}

	void SetRoundingMode(Float32Type::RoundingMode roundingMode) { m_roundingMode = roundingMode; }

	void SetFull(bool full)
	{
		if (full)
		{
			m_roundingMode == Float32Type::RoundingMode::None;
		}
		m_full = full;
	}

	void SetFlush(bool flush) { m_flush = flush; }

	std::string OpCode() const
	{
		if (m_roundingMode == Float32Type::RoundingMode::None)
		{
			return "div" + Float32Type::RoundingModeString(m_roundingMode) + ((m_flush) ? ".ftz" : "") + Float32Type::Name();
		}
		else if (m_full)
		{
			return std::string("div.full") + ((m_flush) ? ".ftz" : "") + Float32Type::Name();
		}
		return std::string("div.approx") + ((m_flush) ? ".ftz" : "") + Float32Type::Name();
	}

	std::string Operands() const
	{
		return m_destination->ToString() + ", " + m_sourceA->ToString() + ", " + m_sourceB->ToString();
	}

private:
	Register<Float32Type> *m_destination = nullptr;
	Operand<Float32Type> *m_sourceA = nullptr;
	Operand<Float32Type> *m_sourceB = nullptr;

	Float32Type::RoundingMode m_roundingMode = Float32Type::RoundingMode::None;
	bool m_full = false;
	bool m_flush = false;
};

template<>
class DivideInstruction<Float64Type> : public PredicatedInstruction
{
public:
	DivideInstruction(Register<Float64Type> *destination, Operand<Float64Type> *sourceA, Operand<Float64Type> *sourceB, Float64Type::RoundingMode roundingMode) : m_destination(destination), m_sourceA(sourceA), m_sourceB(sourceB), m_roundingMode(roundingMode)
	{
		if (roundingMode == Float64Type::RoundingMode::None)
		{
			std::cerr << "PTX::FMAInstruction requires rounding mode with PTX::Float64Type" << std::endl;
			std::exit(EXIT_FAILURE);
		}
	}

	void SetRoundingMode(Float64Type::RoundingMode roundingMode)
	{
		if (roundingMode == Float64Type::RoundingMode::None)
		{
			std::cerr << "PTX::FMAInstruction requires rounding mode with PTX::Float64Type" << std::endl;
			std::exit(EXIT_FAILURE);
		}
		m_roundingMode = roundingMode;
	}

	std::string OpCode() const
	{
		return "div" + Float64Type::RoundingModeString(m_roundingMode) + Float64Type::Name();
	}

	std::string Operands() const
	{
		return m_destination->ToString() + ", " + m_sourceA->ToString() + ", " + m_sourceB->ToString();
	}

private:
	Register<Float64Type> *m_destination = nullptr;
	Operand<Float64Type> *m_sourceA = nullptr;
	Operand<Float64Type> *m_sourceB = nullptr;

	Float64Type::RoundingMode m_roundingMode = Float64Type::RoundingMode::None;
};

}
