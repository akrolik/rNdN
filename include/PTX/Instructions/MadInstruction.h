#pragma once

#include <sstream>

#include "PTX/Statements/InstructionStatement.h"

namespace PTX {

template<class T>
class MadInstruction : public InstructionStatement
{
	static_assert(std::is_base_of<Type, T>::value, "T must be a PTX::Type");
public:
	MadInstruction(Register<T> *dest, Operand<T> *src1, Operand<T> *src2, Operand<T> *src3) : m_destination(dest), m_source1(src1), m_source2(src2), m_source3(src3) {}

	void SetLower(bool lower)
	{
		m_lower = lower;
		if (lower)
		{
			m_upper = false;
		}
	}

	void SetUpper(bool upper)
	{
		m_upper = upper;
		if (upper)
		{
			m_lower = false;
		}
	}

	std::string OpCode()
	{
		std::ostringstream code;

		code << "mad";
		if (m_lower)
		{
			code << ".lo";
		}
		else if (m_upper)
		{
			code << ".hi";
		}
		code << PTX::TypeName<T>();
		return code.str();
	}

	std::string Operands()
	{
		return m_destination->ToString() + ", " + m_source1->ToString() + ", " + m_source2->ToString() + ", " + m_source3->ToString();
	}

private:
	Register<T> *m_destination = nullptr;
	Operand<T> *m_source1 = nullptr;
	Operand<T> *m_source2 = nullptr;
	Operand<T> *m_source3 = nullptr;

	bool m_lower = false;
	bool m_upper = false;
};

}
