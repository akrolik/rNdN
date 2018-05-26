#pragma once

#include "PTX/Statements/InstructionStatement.h"

#include "PTX/Operands/Operand.h"
#include "PTX/Operands/Variables/Register.h"

namespace PTX {

template<class T, Bits B, AddressSpace A>
class ConvertToAddressInstruction : public InstructionStatement
{
	//TODO: Update this instruction
	// DISABLE_BITS(ConvertToAddressInstruction, Bits8);
	// DISABLE_BITS(ConvertToAddressInstruction, Bits16);
public:
	ConvertToAddressInstruction(Register<PointerType<T, B, A>> *destination, Register<PointerType<T, B, AddressSpace::Generic>> *source) : m_destination(destination), m_source(source) {}

	std::string OpCode() const
	{
		return "cvta.to" + AddressSpaceName<A>() + PointerType<T, B, A>::Name();
	}

	std::string Operands() const
	{
		return m_destination->ToString() + ", " + m_source->ToString();
	}

private:
	Register<PointerType<T, B, A>> *m_destination = nullptr;
	Register<PointerType<T, B, AddressSpace::Generic>> *m_source = nullptr;
};

template<class T, AddressSpace A>
using ConvertToAddress32Instruction = ConvertToAddressInstruction<T, Bits::Bits32, A>;
template<class T, AddressSpace A>
using ConvertToAddress64Instruction = ConvertToAddressInstruction<T, Bits::Bits64, A>;

}
