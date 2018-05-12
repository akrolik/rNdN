#pragma once

#include "PTX/Statements/InstructionStatement.h"

#include "PTX/Operands/Address/Address.h"
#include "PTX/Operands/Variable.h"

namespace PTX {

template<Bits A, class T>
class StoreInstruction : public InstructionStatement
{
	static_assert(std::is_base_of<Type, T>::value, "T must be a PTX::Type");
public:
	StoreInstruction(Address<A, T> *address, Register<T> *reg) : m_address(address), m_register(reg) {}

	std::string OpCode() const
	{
		return "st" + GetAddressSpaceName(m_address->GetSpace()) + T::Name();
	}
	
	std::string Operands() const
	{
		return m_address->ToString() + ", " + m_register->ToString();
	}

private:
	Address<A, T> *m_address = nullptr;
	Register<T> *m_register = nullptr;
};

template<class T>
using Store32Instruction = StoreInstruction<Bits::Bits32, T>;
template<class T>
using Store64Instruction = StoreInstruction<Bits::Bits64, T>;

}
