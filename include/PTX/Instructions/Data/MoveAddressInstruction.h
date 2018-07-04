#pragma once

#include "PTX/Instructions/PredicatedInstruction.h"

#include "PTX/Operands/Address/MemoryAddress.h"
#include "PTX/Operands/Variables/Register.h"

namespace PTX {

template<Bits B, class T, class S, bool Assert = true>
class MoveAddressInstruction : public PredicatedInstruction
{
public:
	REQUIRE_TYPE_PARAM(MoveAddressInstruction,
		REQUIRE_BASE(T, DataType)
	);
	REQUIRE_SPACE_PARAM(MoveAddressInstruction,
		REQUIRE_BASE(S, AddressableSpace)
	);

	MoveAddressInstruction(const Register<PointerType<B, T, S>> *destination, const MemoryAddress<B, T, S> *source) : m_destination(destination), m_source(source) {}

	std::string OpCode() const override
	{
		return "mov" + PointerType<B, T, S>::Name();
	}

	std::vector<const Operand *> Operands() const override
	{
		return { m_destination, m_source };
	}

private:
	const Register<PointerType<B, T, S>> *m_destination = nullptr;
	const MemoryAddress<B, T, S> *m_source = nullptr;
};

}
