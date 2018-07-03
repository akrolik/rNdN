#pragma once

#include "PTX/Statements/InstructionStatement.h"

#include "PTX/StateSpace.h"
#include "PTX/Operands/Operand.h"
#include "PTX/Operands/Variables/Register.h"

namespace PTX {

template<Bits B, class T, class S>
class ConvertToAddressInstruction : public InstructionStatement
{
	static_assert(B == Bits::Bits32 || B == Bits::Bits64, "PTX::ConvertToAddressInstruction requires PTX::Bits::Bits32 or PTX::Bits::Bits64");
public:
	ConvertToAddressInstruction(const Register<PointerType<B, T, S>> *destination, const Register<PointerType<B, T>> *source) : m_destination(destination), m_source(source) {}

	std::string OpCode() const override
	{
		return "cvta.to" + S::Name() + PointerType<B, T, S>::Name();
	}

	std::vector<const Operand *> Operands() const override
	{
		return { m_destination, m_source };
	}

private:
	const Register<PointerType<B, T, S>> *m_destination = nullptr;
	const Register<PointerType<B, T>> *m_source = nullptr;
};

template<class T, class S>
using ConvertToAddress32Instruction = ConvertToAddressInstruction<Bits::Bits32, T, S>;
template<class T, class S>
using ConvertToAddress64Instruction = ConvertToAddressInstruction<Bits::Bits64, T, S>;

}
