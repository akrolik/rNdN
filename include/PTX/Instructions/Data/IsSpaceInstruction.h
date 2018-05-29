#pragma once

#include "PTX/Instructions/PredicatedInstruction.h"

#include "PTX/Operands/Address/Address.h"
#include "PTX/Operands/Variables/Register.h"

namespace PTX {

template<Bits B, class T, class S>
class IsSpaceInstruction : public InstructionStatement
{
	REQUIRE_BASE_TYPE(IsSpaceInstruction, Type);
	REQUIRE_BASE_SPACE(IsSpaceInstruction, AddressableSpace);
public:
	IsSpaceInstruction(Register<PredicateType> *destination, Address<B, T> *address) : m_destination(destination), m_address(address) {}

	std::string OpCode() const
	{
		return "isspacep" + S::Name();
	}

	std::string Operands() const
	{
		return m_destination->ToString() + ", " + m_address->ToString();
	}

private:
	Register<PredicateType> *m_destination = nullptr;
	Address<B, T> *m_address = nullptr;
};

template<class T, class S>
using IsSpace32Instruction = IsSpaceInstruction<Bits::Bits32, T, S>;
template<class T, class S>
using IsSpace64Instruction = IsSpaceInstruction<Bits::Bits64, T, S>;

}
