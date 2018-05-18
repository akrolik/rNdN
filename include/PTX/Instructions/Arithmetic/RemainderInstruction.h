#pragma once

#include "PTX/Statements/InstructionStatement.h"

#include "PTX/Operands/Operand.h"
#include "PTX/Operands/Variables/Register.h"

namespace PTX {

template<class T>
class RemainderInstruction : public InstructionStatement
{
	REQUIRE_TYPE(RemainderInstruction, ScalarType);
	DISABLE_TYPE(RemainderInstruction, Int8Type);
	DISABLE_TYPE(RemainderInstruction, UInt8Type);
	DISABLE_TYPES(RemainderInstruction, FloatType);
public:
	RemainderInstruction(Register<T> *destination, Operand<T> *sourceA, Operand<T> *sourceB) : m_destination(destination), m_sourceA(sourceA), m_sourceB(sourceB) {}

	std::string OpCode() const
	{
		return "rem" + T::Name();
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

}
