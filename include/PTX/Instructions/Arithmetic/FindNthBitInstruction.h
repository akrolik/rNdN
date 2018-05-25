#pragma once

#include "PTX/Instructions/PredicatedInstruction.h"

namespace PTX {

template<class T>
class FindNthBitInstruction : public PredicatedInstruction
{
	static_assert(
		std::is_same<Bit32Type, T>::value ||
		std::is_same<Int32Type, T>::value ||
		std::is_same<UInt32Type, T>::value, 
		"PTX::FindNthBitInstruction requires PTX::BitType, PTX::IntType, PTX::UIntType with Bits::Bits32"
	);
public:
	FindNthBitInstruction(Register<Bit32Type> *destination, Operand<Bit32Type> *mask, Operand<T> *base, Operand<Int32Type> *offset) : m_destination(destination), m_mask(mask), m_base(base), m_offset(offset) {}

	std::string OpCode() const
	{
		return "fns" + T::Name();
	}

	std::string Operands() const
	{
		return m_destination->ToString() + ", " + m_mask->ToString() + ", " + m_base->ToString() + ", " + m_offset->ToString();
	}

private:
	Register<Bit32Type> *m_destination = nullptr;
	Operand<Bit32Type> *m_mask = nullptr;
	Operand<T> *m_base = nullptr;
	Operand<Int32Type> *m_offset = nullptr;
};

}
