#pragma once

#include "PTX/Instructions/PredicatedInstruction.h"

#include "PTX/Operands/Operand.h"
#include "PTX/Operands/Variables/Register.h"

namespace PTX {

template<class T>
class MinimumInstruction : public PredicatedInstruction
{
	REQUIRE_TYPE(MinimumInstruction, ScalarType);
	DISABLE_TYPE(MinimumInstruction, Int8Type);
	DISABLE_TYPE(MinimumInstruction, UInt8Type);
	DISABLE_TYPE(MinimumInstruction, Float16Type);
public:
	MinimumInstruction(Register<T> *destination, Operand<T> *sourceA, Operand<T> *sourceB) : m_destination(destination), m_sourceA(sourceA), m_sourceB(sourceB) {}

	std::string OpCode() const
	{
		return "min" + T::Name();
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
class MinimumInstruction<Float32Type> : public PredicatedInstruction
{
public:
	MinimumInstruction(Register<Float32Type> *destination, Operand<Float32Type> *sourceA, Operand<Float32Type> *sourceB) : m_destination(destination), m_sourceA(sourceA), m_sourceB(sourceB) {}

	void SetFlushSubNormal(bool flush) { m_flush = flush; }

	std::string OpCode() const
	{
		if (m_flush)
		{
			return "min.ftz" + Float32Type::Name();
		}
		return "min" + Float32Type::Name();
	}

	std::string Operands() const
	{
		return m_destination->ToString() + ", " + m_sourceA->ToString() + ", " + m_sourceB->ToString();
	}
private:
	Register<Float32Type> *m_destination = nullptr;
	Operand<Float32Type> *m_sourceA = nullptr;
	Operand<Float32Type> *m_sourceB = nullptr;

	bool m_flush = false;
};

}
