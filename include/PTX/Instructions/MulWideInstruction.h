#pragma once

#include "PTX/Statements/InstructionStatement.h"

namespace PTX {

template<class T1, class T2>
class MulWideInstruction : public InstructionStatement
{
	static_assert(std::is_base_of<Type, T1>::value, "T1 must be a PTX::Type");
	static_assert(std::is_base_of<Type, T2>::value, "T2 must be a PTX::Type");
public:
	MulWideInstruction(Register<T1> *dest, Operand<T2> *src1, Operand<T2> *src2) : m_destination(dest), m_source1(src1), m_source2(src2) {}

	std::string OpCode()
	{
		return "mul.wide" + PTX::TypeName<T2>();
	}

	std::string Operands()
	{
		return m_destination->ToString() + ", " + m_source1->ToString() + ", " + m_source2->ToString();
	}

private:
	Register<T1> *m_destination = nullptr;
	Operand<T2> *m_source1 = nullptr;
	Operand<T2> *m_source2 = nullptr;
};

}
