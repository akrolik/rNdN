#pragma once

#include "PTX/Statements/InstructionStatement.h"
#include "PTX/Operands/Register.h"

namespace PTX {

template<Bits B>
class ConverAddressInstruction : public InstructionStatement
{
public:
	ConverAddressInstruction(Register<UIntType<B>> *dest, Register<UIntType<B>> *src) : m_destination(dest), m_source(src) {}

	std::string OpCode()
	{
		//TODO: add state spaces
		return "cvta.to.global" + PTX::TypeName<UIntType<B>>();
	}

	std::string Operands()
	{
		return m_destination->ToString() + ", " + m_source->ToString();
	}

private:
	Register<UIntType<B>> *m_destination = nullptr;
	Register<UIntType<B>> *m_source = nullptr;
};

}
