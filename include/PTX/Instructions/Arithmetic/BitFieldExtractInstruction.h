#pragma once

#include "PTX/Instructions/PredicatedInstruction.h"

namespace PTX {

template<class T>
class BitFieldExtractInstruction : public PredicatedInstruction
{
	REQUIRE_BASE_TYPE(BitFieldExtractInstruction, ScalarType);
	DISABLE_EXACT_TYPE(BitFieldExtractInstruction, Int8Type);
	DISABLE_EXACT_TYPE(BitFieldExtractInstruction, UInt8Type);
	DISABLE_EXACT_TYPE(BitFieldExtractInstruction, Int16Type);
	DISABLE_EXACT_TYPE(BitFieldExtractInstruction, UInt16Type);
	DISABLE_EXACT_TYPE_TEMPLATE(BitFieldExtractInstruction, BitType);
	DISABLE_EXACT_TYPE_TEMPLATE(BitFieldExtractInstruction, FloatType);
public:
	BitFieldExtractInstruction(Register<T> *destination, Operand<T> *sourceA, Operand<UInt32Type> *sourceB, Operand<UInt32Type> *sourceC) : m_destination(destination), m_sourceA(sourceA), m_sourceB(sourceB), m_sourceC(sourceC) {}

	std::string OpCode() const
	{
		return "bfe" + T::Name();
	}

	std::string Operands() const
	{
		return m_destination->ToString() + ", " + m_sourceA->ToString() + ", " + m_sourceB->ToString() + ", " + m_sourceC->ToString();
	}

private:
	Register<T> *m_destination = nullptr;
	Operand<T> *m_sourceA = nullptr;
	Operand<UInt32Type> *m_sourceB = nullptr;
	Operand<UInt32Type> *m_sourceC = nullptr;
};

}
