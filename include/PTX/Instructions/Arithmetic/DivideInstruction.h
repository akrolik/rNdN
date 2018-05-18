#pragma once

#include "PTX/Instructions/InstructionBase.h"

namespace PTX {

template<class T>
class DivideInstruction : public InstructionBase<T, 2>
{
	REQUIRE_TYPE(DivideInstruction, ScalarType);
	DISABLE_TYPE(DivideInstruction, Int8Type);
	DISABLE_TYPE(DivideInstruction, UInt8Type);
	DISABLE_TYPE(DivideInstruction, Float16Type);
public:
	using InstructionBase<T, 2>::InstructionBase;

	std::string OpCode() const
	{
		return "div" + T::Name();
	}
};

template<>
class DivideInstruction<Float32Type> : public InstructionBase<Float32Type, 2>
{
public:
	DivideInstruction(Register<Float32Type> *destination, Operand<Float32Type> *sourceA, Operand<Float32Type> *sourceB, Float32Type::RoundingMode roundingMode = Float32Type::RoundingMode::None) : InstructionBase<Float32Type, 2>(destination, sourceA, sourceB), m_roundingMode(roundingMode) {}

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

private:
	Float32Type::RoundingMode m_roundingMode = Float32Type::RoundingMode::None;
	bool m_full = false;
	bool m_flush = false;
};

template<>
class DivideInstruction<Float64Type> : public InstructionBase<Float64Type, 2>
{
public:
	DivideInstruction(Register<Float64Type> *destination, Operand<Float64Type> *sourceA, Operand<Float64Type> *sourceB, Float64Type::RoundingMode roundingMode) : InstructionBase<Float64Type, 2>(destination, sourceA, sourceB), m_roundingMode(roundingMode)
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

private:
	Float64Type::RoundingMode m_roundingMode = Float64Type::RoundingMode::None;
};

}
