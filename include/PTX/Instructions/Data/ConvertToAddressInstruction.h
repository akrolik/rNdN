#pragma once

#include "PTX/Statements/InstructionStatement.h"

#include "PTX/StateSpace.h"
#include "PTX/Operands/Operand.h"
#include "PTX/Operands/Variables/Register.h"

namespace PTX {

template<class T, Bits B, class S>
class ConvertToAddressInstruction : public InstructionStatement
{
	static_assert(B == Bits::Bits32 || B == Bits::Bits64, "PTX::ConvertToAddressInstruction requires PTX::Bits::Bits32 or PTX::Bits::Bits64");
public:
	ConvertToAddressInstruction(Register<PointerType<T, B, S>> *destination, Register<PointerType<T, B>> *source) : m_destination(destination), m_source(source) {}

	std::string OpCode() const
	{
		return "cvta.to" + S::Name() + PointerType<T, B, S>::Name();
	}

	std::string Operands() const
	{
		return m_destination->ToString() + ", " + m_source->ToString();
	}

private:
	Register<PointerType<T, B, S>> *m_destination = nullptr;
	Register<PointerType<T, B>> *m_source = nullptr;
};

template<class T, class S>
using ConvertToAddress32Instruction = ConvertToAddressInstruction<T, Bits::Bits32, S>;
template<class T, class S>
using ConvertToAddress64Instruction = ConvertToAddressInstruction<T, Bits::Bits64, S>;

}
