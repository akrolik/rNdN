#pragma once

#include "PTX/Statements/InstructionStatement.h"

#include "PTX/Operands/Address/Address.h"
#include "PTX/Operands/Variable.h"

namespace PTX {

template<Bits A, class T>
class LoadInstruction : public InstructionStatement
{
	static_assert(std::is_base_of<Type, T>::value, "T must be a PTX::Type");
public:
	LoadInstruction(Register<T> *reg, Address<A, T> *address) : m_register(reg), m_address(address) {}

	std::string OpCode() const
	{
		return "ld" + GetAddressSpaceName(m_address->GetSpace()) + T::Name();
	}
	
	std::string Operands() const
	{
		return m_register->ToString() + ", " + m_address->ToString();
	}

private:
	Register<T> *m_register = nullptr;
	Address<A, T> *m_address = nullptr;
};

template<class T>
using Load32Instruction = LoadInstruction<Bits::Bits32, T>;
template<class T>
using Load64Instruction = LoadInstruction<Bits::Bits64, T>;

}
