#pragma once

#include "PTX/Instructions/PredicatedInstruction.h"

#include "PTX/Operands/Operand.h"
#include "PTX/Operands/Variables/Register.h"

namespace PTX {

template<class T>
class MaximumInstruction : public PredicatedInstruction
{
	REQUIRE_TYPE(MaximumInstruction, ScalarType);
	DISABLE_TYPE(MaximumInstruction, Int8Type);
	DISABLE_TYPE(MaximumInstruction, UInt8Type);
	DISABLE_TYPE(MaximumInstruction, Float16Type);
public:
	MaximumInstruction(Register<T> *destination, Operand<T> *sourceA, Operand<T> *sourceB) : m_destination(destination), m_sourceA(sourceA), m_sourceB(sourceB) {}

	std::string OpCode() const
	{
		return "max" + T::Name();
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
class MaximumInstruction<Float32Type> : public PredicatedInstruction
{
public:
	MaximumInstruction(Register<Float32Type> *destination, Operand<Float32Type> *sourceA, Operand<Float32Type> *sourceB) : m_destination(destination), m_sourceA(sourceA), m_sourceB(sourceB) {}

	void SetFlushSubNormal(bool flush) { m_flush = flush; }

	std::string OpCode() const
	{
		if (m_flush)
		{
			return "max.ftz" + Float32Type::Name();
		}
		return "max" + Float32Type::Name();
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
