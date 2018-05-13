#pragma once

#include "PTX/Statements/InstructionStatement.h"

#include "PTX/Operands/Variables/Register.h"

namespace PTX {

template<class T>
class MoveInstruction : public InstructionStatement
{
	static_assert(std::is_base_of<ValueType, T>::value, "T must be a PTX::ValueType");
public:
	MoveInstruction(Register<T> *destination, Register<T> *source) : m_destination(destination), m_source(source) {}

	std::string OpCode() const
	{
		return "mov" + T::Name();
	}

	std::string Operands() const
	{
		return m_destination->ToString() + ", " + m_source->ToString();
	}

private:
	Register<T> *m_destination = nullptr;
	Register<T> *m_source = nullptr;
};

}
