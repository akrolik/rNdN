#pragma once

#include "PTX/Statements/InstructionStatement.h"

namespace PTX {

template<class T>
class AddInstruction : public InstructionStatement
{
	static_assert(std::is_base_of<Type, T>::value, "T must be a PTX::Type");
public:
	AddInstruction(Register<T> *dest, Operand<T> *src1, Operand<T> *src2) : m_destination(dest), m_source1(src1), m_source2(src2) {}

	std::string OpCode()
	{
		return "add" + PTX::TypeName<T>();
	}

	std::string Operands()
	{
		return m_destination->ToString() + ", " + m_source1->ToString() + ", " + m_source2->ToString();
	}

private:
	Register<T> *m_destination = nullptr;
	Operand<T> *m_source1 = nullptr;
	Operand<T> *m_source2 = nullptr;
};

}
