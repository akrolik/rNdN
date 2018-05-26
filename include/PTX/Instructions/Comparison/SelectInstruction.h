#pragma once

#include <sstream>

#include "PTX/Instructions/PredicatedInstruction.h"

#include "PTX/Operands/Operand.h"
#include "PTX/Operands/Variables/Register.h"

namespace PTX {

template<class T>
class SelectInstruction : public PredicatedInstruction
{
	REQUIRE_BASE_TYPE(SelectInstruction, ScalarType);
	DISABLE_EXACT_TYPE(SelectInstruction, PredicateType);
	DISABLE_EXACT_TYPE(SelectInstruction, Bit8Type);
	DISABLE_EXACT_TYPE(SelectInstruction, Int8Type);
	DISABLE_EXACT_TYPE(SelectInstruction, UInt8Type);
	DISABLE_EXACT_TYPE(SelectInstruction, Float16Type);
	DISABLE_EXACT_TYPE(SelectInstruction, Float16x2Type);
public:
	SelectInstruction(Register<T> *destination, Operand<T> *sourceA, Operand<T> *sourceB, Operand<PredicateType> *sourceC) : m_destination(destination), m_sourceA(sourceA), m_sourceB(sourceB), m_sourceC(sourceC) {}

	std::string OpCode() const
	{
		return "selp" + T::Name();
	}

	std::string Operands() const
	{
		return m_destination->ToString() + ", " + m_sourceA->ToString() + ", " + m_sourceB->ToString() + ", " + m_sourceC->ToString();
	}

private:
	Register<T> *m_destination = nullptr;
	Operand<T> *m_sourceA = nullptr;
	Operand<T> *m_sourceB = nullptr;
	Register<PredicateType> *m_sourceC = nullptr;
};

}
