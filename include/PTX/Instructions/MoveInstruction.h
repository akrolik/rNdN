#pragma once

#include "PTX/Statements/InstructionStatement.h"
#include "PTX/Operands/Register.h"

namespace PTX {

template<class T>
class MoveInstruction : public InstructionStatement
{
	static_assert(std::is_base_of<Type, T>::value, "T must be a PTX::Type");
public:
	MoveInstruction(Register<T> *destination, Register<T> *source) : m_destination(destination), m_source(source) {}

	std::string OpCode()
	{
		return "mov" + PTX::TypeName<T>();
	}

	std::string Operands()
	{
		return m_destination->ToString() + ", " + m_source->ToString();
	}

private:
	Register<T> *m_destination = nullptr;
	Register<T> *m_source = nullptr;
};

}
