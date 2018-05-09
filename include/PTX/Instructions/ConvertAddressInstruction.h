#pragma once

#include "PTX/Statements/InstructionStatement.h"
#include "PTX/Operands/Register/Register.h"

namespace PTX {

template<Bits B>
class ConvertAddressInstruction : public InstructionStatement
{
public:
	ConvertAddressInstruction(Register<UIntType<B>> *destination, Register<UIntType<B>> *source) : m_destination(destination), m_source(source) {}

	std::string OpCode() const
	{
		//TODO: add state spaces
		return "cvta.to.global" + PTX::TypeName<UIntType<B>>();
	}

	std::string Operands() const
	{
		return m_destination->ToString() + ", " + m_source->ToString();
	}

private:
	Register<UIntType<B>> *m_destination = nullptr;
	Register<UIntType<B>> *m_source = nullptr;
};

using ConvertAddress32Instruction = ConvertAddressInstruction<Bits::Bits32>;
using ConvertAddress64Instruction = ConvertAddressInstruction<Bits::Bits64>;

}
