#pragma once

#include "PTX/Instructions/PredicatedInstruction.h"

namespace PTX {

template<class T>
class ShiftRightInstruction : public PredicatedInstruction
{
	REQUIRE_EXACT_TYPE_TEMPLATE(ShiftRightInstruction, BitType);
	DISABLE_EXACT_TYPE(ShiftRightInstruction, PredicateType);
	DISABLE_EXACT_TYPE(ShiftRightInstruction, Bit8Type);
public:
	ShiftRightInstruction(Register<T> *destination, Operand<T> *source, Operand<UInt32Type> *shift) : m_destination(destination), m_source(source), m_shift(shift) {}

	std::string OpCode() const
	{
		return "shr" + T::Name();
	}
	
	std::string Operands() const
	{
		return m_destination->ToString() + ", " + m_source->ToString() + ", " + m_shift->ToString();
	}

private:
	Register<T> *m_destination = nullptr;
	Operand<T> *m_source = nullptr;
	Operand<UInt32Type> *m_shift = nullptr;
};

}
