#pragma once

#include "PTX/Statements/InstructionStatement.h"

#include "PTX/Operands/Operand.h"
#include "PTX/Operands/Variables/Register.h"

namespace PTX {

template<Bits B, AddressSpace S>
class ConvertToAddressInstruction : public InstructionStatement
{
	DISABLE_BITS(ConvertToAddressInstruction, Bits8);
	DISABLE_BITS(ConvertToAddressInstruction, Bits16);
public:
	ConvertToAddressInstruction(Register<UIntType<B>> *destination, Register<UIntType<B>> *source) : m_destination(destination), m_source(source) {}

	std::string OpCode() const
	{
		return "cvta.to" + AddressSpaceName<S>() + UIntType<B>::Name();
	}

	std::string Operands() const
	{
		return m_destination->ToString() + ", " + m_source->ToString();
	}

private:
	Register<UIntType<B>> *m_destination = nullptr;
	Register<UIntType<B>> *m_source = nullptr;
};

template<AddressSpace S>
using ConvertToAddress32Instruction = ConvertToAddressInstruction<Bits::Bits32, S>;
template<AddressSpace S>
using ConvertToAddress64Instruction = ConvertToAddressInstruction<Bits::Bits64, S>;

}
