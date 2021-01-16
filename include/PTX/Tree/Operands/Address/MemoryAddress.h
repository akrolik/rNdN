#pragma once

#include "PTX/Tree/Operands/Address/Address.h"

namespace PTX {

DispatchInterface_Data(MemoryAddress)

template<Bits B, class T, class S = AddressableSpace, bool Assert = true>
class MemoryAddress : DispatchInherit(MemoryAddress), public Address<B, T, S, Assert>
{
public:
	REQUIRE_SPACE_PARAM(MemoryAddress,
		REQUIRE_BASE(S, AddressableSpace) && !REQUIRE_EXACT(S, AddressableSpace)
	);

	MemoryAddress(Variable<T, S> *variable, int offset = 0) : m_variable(variable), m_offset(offset) {}

	// Properties

	const Variable<T, S> *GetVariable() const { return m_variable; }
	Variable<T, S> *GetVariable() { return m_variable; }
	void SetVariable(Variable<T, S> *variable) { m_variable = variable; }

	int GetOffset() const { return m_offset; }
	void SetOffset(int offset) { m_offset = offset; }

	MemoryAddress<B, T, S, Assert> *CreateOffsetAddress(int offset) const override
	{
		return new MemoryAddress(m_variable, m_offset + offset);
	}

	// Formatting

	std::string ToString() const override
	{
		if (m_offset != 0)
		{
			return m_variable->ToString() + "+" + std::to_string(static_cast<int>(sizeof(typename T::SystemType)) * m_offset);
		}
		else
		{
			return m_variable->ToString();
		}
	}

	json ToJSON() const override
	{
		json j;
		j["kind"] = "PTX::MemoryAddress";
		j["variable"] = m_variable->ToJSON();
		if (m_offset != 0)
		{
			j["offset"] = m_offset;
		}
		return j;
	}

	// Visitors

	void Accept(OperandVisitor& visitor) override { visitor.Visit(this); }
	void Accept(ConstOperandVisitor& visitor) const override { visitor.Visit(this); }

protected:
	DispatchMember_Bits(B);
	DispatchMember_Type(T);
	DispatchMember_Space(S);

	Variable<T, S> *m_variable = nullptr;
	int m_offset = 0;
};

template<class T, class S = AddressableSpace>
using MemoryAddress32 = MemoryAddress<Bits::Bits32, T, S>;
template<class T, class S = AddressableSpace>
using MemoryAddress64 = MemoryAddress<Bits::Bits64, T, S>;

}
