#pragma once

#include "PTX/Statements/InstructionStatement.h"

#include "PTX/Operands/Address/Address.h"
#include "PTX/Operands/Variables/Register.h"

namespace PTX {

template<Bits A, class T, AddressSpace S>
class LoadInstruction : public InstructionStatement
{
	static_assert(std::is_base_of<Type, T>::value, "T must be a PTX::Type");
public:
	LoadInstruction(Register<T> *reg, Address<A, T, S> *address) : m_register(reg), m_address(address) {}

	std::string OpCode() const
	{
		return "ld" + AddressSpaceName<S>() + T::Name();
	}
	
	std::string Operands() const
	{
		return m_register->ToString() + ", " + m_address->ToString();
	}

private:
	Register<T> *m_register = nullptr;
	Address<A, T, S> *m_address = nullptr;
};

template<class T, AddressSpace S>
using Load32Instruction = LoadInstruction<Bits::Bits32, T, S>;
template<class T, AddressSpace S>
using Load64Instruction = LoadInstruction<Bits::Bits64, T, S>;

}
