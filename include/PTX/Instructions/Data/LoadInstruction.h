#pragma once

#include "PTX/Instructions/PredicatedInstruction.h"

#include "PTX/Operands/Address/Address.h"
#include "PTX/Operands/Variables/Register.h"

namespace PTX {

template<Bits B, class T, AddressSpace A>
class LoadInstruction : public PredicatedInstruction
{
	REQUIRE_TYPE(LoadInstruction, DataType);
	DISABLE_TYPE(LoadInstruction, Float16Type);
public:
	LoadInstruction(Register<T> *reg, Address<B, T, A> *address) : m_register(reg), m_address(address) {}

	std::string OpCode() const
	{
		return "ld" + AddressSpaceName<A>() + T::Name();
	}
	
	std::string Operands() const
	{
		return m_register->ToString() + ", " + m_address->ToString();
	}

private:
	Register<T> *m_register = nullptr;
	Address<B, T, A> *m_address = nullptr;
};

template<class T, AddressSpace A>
using Load32Instruction = LoadInstruction<Bits::Bits32, T, A>;
template<class T, AddressSpace A>
using Load64Instruction = LoadInstruction<Bits::Bits64, T, A>;

}
