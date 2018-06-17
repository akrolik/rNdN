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
	IsSpaceInstruction(const Register<PredicateType> *destination, const Address<B, T> *address) : m_destination(destination), m_address(address) {}

	std::string OpCode() const override
	{
		return "isspacep" + S::Name();
	}

	std::string Operands() const override
	{
		return m_destination->ToString() + ", " + m_address->ToString();
	}

private:
	const Register<PredicateType> *m_destination = nullptr;
	const Address<B, T> *m_address = nullptr;
};

template<class T, class S>
using IsSpace32Instruction = IsSpaceInstruction<Bits::Bits32, T, S>;
template<class T, class S>
using IsSpace64Instruction = IsSpaceInstruction<Bits::Bits64, T, S>;

}
