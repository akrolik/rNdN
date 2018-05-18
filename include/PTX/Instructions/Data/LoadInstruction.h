#pragma once

#include "PTX/Instructions/PredicatedInstruction.h"

#include "PTX/Operands/Address/Address.h"
#include "PTX/Operands/Variables/Register.h"

namespace PTX {

template<Bits A, class T, AddressSpace S>
class LoadInstruction : public PredicatedInstruction
{
	REQUIRE_TYPE(LoadInstruction, DataType);
	DISABLE_TYPE(LoadInstruction, Float16Type);
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

template<Bits A, class T>
class LoadInstruction<A, T, Param> : public PredicatedInstruction
{
	REQUIRE_TYPE(LoadInstruction, DataType);
	DISABLE_TYPE(LoadInstruction, Float16Type);
public:
	LoadInstruction(Register<UIntType<A>> *reg, Address<A, T, Param> *address) : m_register(reg), m_address(address) {}

	std::string OpCode() const
	{
		return "ld.param.u" + std::to_string(A);
	}

	std::string Operands() const
	{
		return m_register->ToString() + ", " + m_address->ToString();
	}

private:
	Register<UIntType<A>> *m_register = nullptr;
	Address<A, T, Param> *m_address = nullptr;
};

template<class T, AddressSpace S>
using Load32Instruction = LoadInstruction<Bits::Bits32, T, S>;
template<class T, AddressSpace S>
using Load64Instruction = LoadInstruction<Bits::Bits64, T, S>;

}
