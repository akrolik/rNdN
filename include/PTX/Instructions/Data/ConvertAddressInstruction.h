#pragma once

#include "PTX/Statements/InstructionStatement.h"

#include "PTX/Operands/Operand.h"
#include "PTX/Operands/Variables/Register.h"

namespace PTX {

//TODO: add variable source

template<class T, Bits B, AddressSpace A>
class ConvertAddressInstruction : public InstructionStatement
{
	static_assert(B == Bits::Bits32 || B == Bits::Bits64, "PTX::ConvertAddressInstruction requires PTX::Bits::Bits32 or PTX::Bits::Bits64");
public:
	ConvertAddressInstruction(Register<PointerType<T, B>> *destination, Register<PointerType<T, B, A>> *source) : m_destination(destination), m_source(source) {}

	std::string OpCode() const
	{
		return "cvta" + AddressSpaceName<A>() + PointerType<T, B, A>::Name();
	}

	std::string Operands() const
	{
		return m_destination->String() + ", " + m_source->String();
	}

private:
	Register<PointerType<T, B>> *m_destination = nullptr;
	Register<PointerType<T, B, A>> *m_source = nullptr;
};

template<class T, AddressSpace A>
using ConvertAddress32Instruction = ConvertAddressInstruction<T, Bits::Bits32, A>;
template<class T, AddressSpace A>
using ConvertAddress64Instruction = ConvertAddressInstruction<T, Bits::Bits64, A>;

}
