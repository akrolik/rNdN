#pragma once

#include "PTX/Statements/InstructionStatement.h"
#include "PTX/Operands/Register.h"
#include "PTX/Operands/Address.h"

namespace PTX {

template<Bits A, class T, VectorSize V = Scalar>
class StoreInstruction : public InstructionStatement
{
	static_assert(std::is_base_of<Type, T>::value, "T must be a PTX::Type");
public:
	StoreInstruction(Address<A, T, V> *address, Register<T, V> *reg) : m_address(address), m_register(reg) {}

	std::string OpCode()
	{
		return "st" + m_address->SpaceName() + PTX::TypeName<T>();
	}
	
	std::string Operands()
	{
		return m_address->ToString() + ", " + m_register->ToString();
	}

private:
	Address<A, T, V> *m_address = nullptr;
	Register<T, V> *m_register = nullptr;
};

}
