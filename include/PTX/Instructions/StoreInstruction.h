#pragma once

#include "PTX/Statements/InstructionStatement.h"
#include "PTX/Operands/Register/Register.h"
#include "PTX/Operands/Address/Address.h"

namespace PTX {

template<Bits A, class T, VectorSize V = Scalar>
class StoreInstruction : public InstructionStatement
{
	static_assert(std::is_base_of<Type, T>::value, "T must be a PTX::Type");
public:
	StoreInstruction(Address<A, T, V> *address, Register<T, V> *reg) : m_address(address), m_register(reg) {}

	std::string OpCode() const
	{
		return "st" + GetAddressSpaceName(m_address->GetSpace()) + PTX::TypeName<T>();
	}
	
	std::string Operands() const
	{
		return m_address->ToString() + ", " + m_register->ToString();
	}

private:
	Address<A, T, V> *m_address = nullptr;
	Register<T, V> *m_register = nullptr;
};

template<class T, VectorSize V = Scalar>
using Store32Instruction = StoreInstruction<Bits::Bits32, T, V>;
template<class T, VectorSize V = Scalar>
using Store64Instruction = StoreInstruction<Bits::Bits64, T, V>;

}
