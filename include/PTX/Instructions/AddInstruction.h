#pragma once

#include "PTX/Statements/InstructionStatement.h"

#include "PTX/Operands/Operand.h"
#include "PTX/Operands/Variables/Register.h"

namespace PTX {

template<class T>
class AddInstruction : public InstructionStatement
{
	static_assert(std::is_base_of<ValueType, T>::value, "T must be a PTX::ValueType");
public:
	AddInstruction(Register<T> *destination, Operand<T> *sourceA, Operand<T> *sourceB) : m_destination(destination), m_sourceA(sourceA), m_sourceB(sourceB) {}

	std::string OpCode() const
	{
		return "add" + T::Name();
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
