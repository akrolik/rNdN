#pragma once

#include "PTX/Instructions/PredicatedInstruction.h"

namespace PTX {

template<class T>
class ShiftLeftInstruction : public PredicatedInstruction
{
	REQUIRE_EXACT_TYPE_TEMPLATE(ShiftLeftInstruction, BitType);
	DISABLE_EXACT_TYPE(ShiftLeftInstruction, PredicateType);
	DISABLE_EXACT_TYPE(ShiftLeftInstruction, Bit8Type);
public:
	ShiftLeftInstruction(Register<T> *destination, Operand<T> *source, Operand<UInt32Type> *shift) : m_destination(destination), m_source(source), m_shift(shift) {}

	std::string OpCode() const
	{
		return "shl" + T::Name();
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
