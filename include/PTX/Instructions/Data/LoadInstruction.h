#pragma once

#include "PTX/Instructions/PredicatedInstruction.h"

#include "PTX/StateSpace.h"
#include "PTX/Operands/Address/Address.h"
#include "PTX/Operands/Variables/Register.h"

namespace PTX {

template<Bits B, class T, class S>
class LoadInstruction : public PredicatedInstruction
{
	REQUIRE_BASE_TYPE(LoadInstruction, DataType);
	DISABLE_EXACT_TYPE(LoadInstruction, Float16Type);
public:
	LoadInstruction(Register<T> *reg, Address<B, T, S> *address) : m_register(reg), m_address(address) {}

	std::string OpCode() const
	{
		return "ld" + S::Name() + T::Name();
	}
	
	std::string Operands() const
	{
		return m_register->ToString() + ", " + m_address->ToString();
	}

private:
	Register<T> *m_register = nullptr;
	Address<B, T, S> *m_address = nullptr;
};

template<class T, class S>
using Load32Instruction = LoadInstruction<Bits::Bits32, T, S>;
template<class T, class S>
using Load64Instruction = LoadInstruction<Bits::Bits64, T, S>;

}
