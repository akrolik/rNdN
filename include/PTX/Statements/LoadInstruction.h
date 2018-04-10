#pragma once

#include "PTX/Statements/InstructionStatement.h"
#include "PTX/Operands/Register.h"
#include "PTX/Operands/Address.h"

namespace PTX {

template<Type T, VectorSize V = Scalar>
class LoadInstruction : public InstructionStatement
{
public:
	LoadInstruction(Register<T, V> *reg, Address<T, V> *address) : m_register(reg), m_address(address) {}

	std::string OpCode()
	{
		return "ld" + m_address->Space()->SpaceName() + PTX::TypeName<T>();
	}
	
	std::string Operands()
	{
		return m_register->ToString() + ", " + m_address->ToString();
	}

private:
	Register<T, V> *m_register = nullptr;
	Address<T, V> *m_address = nullptr;
};

}
