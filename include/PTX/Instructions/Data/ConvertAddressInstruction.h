#pragma once

#include "PTX/Statements/InstructionStatement.h"

#include "PTX/Operands/Operand.h"
#include "PTX/Operands/Variables/Register.h"

namespace PTX {

//TODO: add variable source

template<class T, Bits B, class S>
class ConvertAddressInstruction : public InstructionStatement
{
	static_assert(B == Bits::Bits32 || B == Bits::Bits64, "PTX::ConvertAddressInstruction requires PTX::Bits::Bits32 or PTX::Bits::Bits64");
public:
	ConvertAddressInstruction(const Register<PointerType<T, B>> *destination, const Register<PointerType<T, B, S>> *source) : m_destination(destination), m_source(source) {}

	std::string OpCode() const override
	{
		return "cvta" + S::Name() + PointerType<T, B, S>::Name();
	}

	std::string Operands() const override
	{
		return m_destination->String() + ", " + m_source->String();
	}

private:
	const Register<PointerType<T, B>> *m_destination = nullptr;
	const Register<PointerType<T, B, S>> *m_source = nullptr;
};

template<class T, class S>
using ConvertAddress32Instruction = ConvertAddressInstruction<T, Bits::Bits32, S>;
template<class T, class S>
using ConvertAddress64Instruction = ConvertAddressInstruction<T, Bits::Bits64, S>;

}
