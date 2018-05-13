#pragma once

#include "PTX/Statements/InstructionStatement.h"

#include "PTX/Operands/Operand.h"
#include "PTX/Operands/Variables/Register.h"

namespace PTX {

template<class T1, class T2>
class MulWideInstruction : public InstructionStatement
{
	static_assert(std::is_base_of<Type, T1>::value, "T1 must be a PTX::Type");
	static_assert(std::is_base_of<Type, T2>::value, "T2 must be a PTX::Type");
public:
	MulWideInstruction(Register<T1> *destination, Operand<T2> *sourceA, Operand<T2> *sourceB) : m_destination(destination), m_sourceA(sourceA), m_sourceB(sourceB) {}

	std::string OpCode() const
	{
		return "mul.wide" + T2::Name();
	}

	std::string Operands() const
	{
		return m_destination->ToString() + ", " + m_sourceA->ToString() + ", " + m_sourceB->ToString();
	}

private:
	Register<T1> *m_destination = nullptr;
	Operand<T2> *m_sourceA = nullptr;
	Operand<T2> *m_sourceB = nullptr;
};

}
