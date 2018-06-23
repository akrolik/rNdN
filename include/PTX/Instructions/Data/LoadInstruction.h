#pragma once

#include "PTX/Instructions/PredicatedInstruction.h"

#include "PTX/StateSpace.h"
#include "PTX/Operands/Address/Address.h"
#include "PTX/Operands/Variables/Register.h"

namespace PTX {

template<Bits B, class T, class S>
class LoadInstruction : public PredicatedInstruction
{
	// REQUIRE_BASE_TYPE(LoadInstruction, DataType);
	// DISABLE_EXACT_TYPE(LoadInstruction, Float16Type);
public:
	LoadInstruction(const Register<T> *reg, const Address<B, T, S> *address) : m_register(reg), m_address(address) {}

	std::string OpCode() const override
	{
		return "ld" + S::Name() + T::Name();
	}
	
	std::vector<const Operand *> Operands() const override
	{
		return { m_register, m_address };
	}

private:
	const Register<T> *m_register = nullptr;
	const Address<B, T, S> *m_address = nullptr;
};

template<class T, class S>
using Load32Instruction = LoadInstruction<Bits::Bits32, T, S>;
template<class T, class S>
using Load64Instruction = LoadInstruction<Bits::Bits64, T, S>;

}
