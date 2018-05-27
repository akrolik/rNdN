#pragma once

#include "PTX/Instructions/PredicatedInstruction.h"

#include "PTX/Operands/Address/Address.h"
#include "PTX/Operands/Variables/Register.h"

namespace PTX {

template<Bits B, class T, AddressSpace A>
class IsSpaceInstruction : public InstructionStatement
{
public:
	IsSpaceInstruction(Register<PredicateType> *destination, Address<B, T> *address) : m_destination(destination), m_address(address) {}

	std::string OpCode() const
	{
		return "isspacep" + AddressSpaceName<A>();
	}

	std::string Operands() const
	{
		return m_destination->ToString() + ", " + m_address->ToString();
	}

private:
	Register<PredicateType> *m_destination = nullptr;
	Address<B, T> *m_address = nullptr;
};

template<class T, AddressSpace A>
using IsSpace32Instruction = IsSpaceInstruction<Bits::Bits32, T, A>;
template<class T, AddressSpace A>
using IsSpace64Instruction = IsSpaceInstruction<Bits::Bits64, T, A>;

}
