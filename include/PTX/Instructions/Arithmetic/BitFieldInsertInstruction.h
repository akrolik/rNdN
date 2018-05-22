#pragma once

#include "PTX/Instructions/PredicatedInstruction.h"

namespace PTX {

template<class T>
class BitFieldInsertInstruction : public PredicatedInstruction
{
	REQUIRE_TYPES(BitFieldInsertInstruction, BitType);
	DISABLE_TYPE(BitFieldInsertInstruction, Bit8Type);
	DISABLE_TYPE(BitFieldInsertInstruction, Bit16Type);
public:
	BitFieldInsertInstruction(Register<T> *destination, Operand<T> *sourceA, Operand<T> *sourceB, Operand<UInt32Type> *sourceC, Operand<UInt32Type> *sourceD) : m_destination(destination), m_sourceA(sourceA), m_sourceB(sourceB), m_sourceC(sourceC), m_sourceD(sourceD) {}

	std::string OpCode() const
	{
		return "bfi" + T::Name();
	}

	std::string Operands() const
	{
		return m_destination->ToString() + ", " + m_sourceA->ToString() + ", " + m_sourceB->ToString() + ", " + m_sourceC->ToString() + ", " + m_sourceD->ToString();
	}

private:
	Register<T> *m_destination = nullptr;
	Operand<T> *m_sourceA = nullptr;
	Operand<T> *m_sourceB = nullptr;
	Operand<UInt32Type> *m_sourceC = nullptr;
	Operand<UInt32Type> *m_sourceD = nullptr;
};

}
