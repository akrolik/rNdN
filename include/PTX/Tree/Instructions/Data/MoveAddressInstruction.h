#pragma once

#include "PTX/Tree/Instructions/PredicatedInstruction.h"

#include "PTX/Tree/Operands/Address/MemoryAddress.h"
#include "PTX/Tree/Operands/Variables/Register.h"

namespace PTX {

template<Bits B, class T, class S, bool Assert = true>
class MoveAddressInstruction : public PredicatedInstruction
{
public:
	REQUIRE_TYPE_PARAM(MoveAddressInstruction,
		REQUIRE_BASE(T, ValueType)
	);
	REQUIRE_SPACE_PARAM(MoveAddressInstruction,
		REQUIRE_BASE(S, AddressableSpace)
	);

	MoveAddressInstruction(const Register<PointerType<B, T, S>> *destination, const MemoryAddress<B, T, S> *source) : m_destination(destination), m_source(source) {}

	const Register<PointerType<B, T, S>> *GetDestination() const { return m_destination; }
	void SetDestination(Register<PointerType<B, T, S>> *destination) { m_destination = destination; }

	const MemoryAddress<B, T, S> *GetAddress() const { return m_source; }
	void SetAddress(const MemoryAddress<B, T, S> *address) { m_source = address; }

	static std::string Mnemonic() { return "mov"; }

	std::string OpCode() const override
	{
		return Mnemonic() + PointerType<B, T, S>::Name();
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
