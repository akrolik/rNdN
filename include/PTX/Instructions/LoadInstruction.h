#pragma once

#include "PTX/Statements/InstructionStatement.h"
#include "PTX/Operands/Register.h"
#include "PTX/Operands/Address.h"

namespace PTX {

template<Bits A, class T, VectorSize V = Scalar>
class LoadInstruction : public InstructionStatement
{
	static_assert(std::is_base_of<Type, T>::value, "T must be a PTX::Type");
public:
	LoadInstruction(Register<T, V> *reg, Address<A, T, V> *address) : m_register(reg), m_address(address) {}

	std::string OpCode()
	{
		return "ld" + m_address->SpaceName() + PTX::TypeName<T>();
	}
	
	std::string Operands()
	{
		return m_register->ToString() + ", " + m_address->ToString();
	}

private:
	Register<T, V> *m_register = nullptr;
	Address<A, T, V> *m_address = nullptr;
};

template<class T, VectorSize V = Scalar>
using Load32Instruction = LoadInstruction<Bits::Bits32, T, V>;
template<class T, VectorSize V = Scalar>
using Load64Instruction = LoadInstruction<Bits::Bits64, T, V>;

}
