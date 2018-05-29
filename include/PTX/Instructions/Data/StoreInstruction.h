#pragma once

#include "PTX/Statements/InstructionStatement.h"

#include "PTX/StateSpace.h"
#include "PTX/Operands/Address/Address.h"
#include "PTX/Operands/Variables/Register.h"

namespace PTX {

template<Bits A, class T, class S>
class StoreInstruction : public InstructionStatement
{
	REQUIRE_BASE_TYPE(StoreInstruction, DataType);
	DISABLE_EXACT_TYPE(StoreInstruction, Float16Type);
public:
	StoreInstruction(Address<A, T, S> *address, Register<T> *reg) : m_address(address), m_register(reg) {}

	std::string OpCode() const
	{
		return "st" + S::Name() + T::Name();
	}
	
	std::string Operands() const
	{
		return m_address->ToString() + ", " + m_register->ToString();
	}

private:
	Address<A, T, S> *m_address = nullptr;
	Register<T> *m_register = nullptr;
};

template<class T, class S>
using Store32Instruction = StoreInstruction<Bits::Bits32, T, S>;
template<class T, class S>
using Store64Instruction = StoreInstruction<Bits::Bits64, T, S>;

}
