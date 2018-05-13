#pragma once

#include <sstream>

#include "PTX/Statements/InstructionStatement.h"

#include "PTX/Operands/Operand.h"
#include "PTX/Operands/Variables/Register.h"

namespace PTX {

template<class T>
class MadInstruction : public InstructionStatement
{
	static_assert(std::is_base_of<ValueType, T>::value, "T must be a PTX::ValueType");
public:
	MadInstruction(Register<T> *destination, Operand<T> *sourceA, Operand<T> *sourceB, Operand<T> *sourceC) : m_destination(destination), m_sourceA(sourceA), m_sourceB(sourceB), m_sourceC(sourceC) {}

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

	std::string OpCode() const
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
		code << T::Name();
		return code.str();
	}

	std::string Operands() const
	{
		return m_destination->ToString() + ", " + m_sourceA->ToString() + ", " + m_sourceB->ToString() + ", " + m_sourceC->ToString();
	}

private:
	Register<T> *m_destination = nullptr;
	Operand<T> *m_sourceA = nullptr;
	Operand<T> *m_sourceB = nullptr;
	Operand<T> *m_sourceC = nullptr;

	bool m_lower = false;
	bool m_upper = false;
};

}
