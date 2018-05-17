#pragma once

#include "PTX/Instructions/PredicatedInstruction.h"

#include "PTX/Operands/Operand.h"
#include "PTX/Operands/Variables/Register.h"

namespace PTX {

template<class T1, class T2>
class MADWideInstruction : public PredicatedInstruction
{
	static_assert(
		(std::is_same<Int64Type, T1>::value && std::is_same<Int32Type, T2>::value) ||
		(std::is_same<UInt64Type, T1>::value && std::is_same<UInt32Type, T2>::value),
		"T1 and T2 must be 16-, 32-, or 64-bit integers (signed or unsigned), with T1 = 2x T2"
	);
public:
	MADWideInstruction(Register<T1> *destination, Operand<T2> *sourceA, Operand<T2> *sourceB, Operand<T2> *sourceC) : m_destination(destination), m_sourceA(sourceA), m_sourceB(sourceB), m_sourceC(sourceC) {}

	std::string OpCode() const
	{
		return "mad.wide" + T2::Name();
	}

	std::string Operands() const
	{
		return m_destination->ToString() + ", " + m_sourceA->ToString() + ", " + m_sourceB->ToString() + ", " + m_sourceC->ToString();
	}

private:
	Register<T1> *m_destination = nullptr;
	Operand<T2> *m_sourceA = nullptr;
	Operand<T2> *m_sourceB = nullptr;
	Operand<T2> *m_sourceC = nullptr;
};

}
