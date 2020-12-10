#pragma once

#include "PTX/Tree/Operands/Address/Address.h"

namespace PTX {

template<Bits B, class T, class S = AddressableSpace>
class MemoryAddress : public Address<B, T, S>
{
public:
	REQUIRE_TYPE_PARAM(MemoryAddress,
		REQUIRE_BASE(T, ValueType)
	);
	REQUIRE_SPACE_PARAM(MemoryAddress,
		REQUIRE_BASE(S, AddressableSpace)
	);

	MemoryAddress(typename S::template VariableType<T> *variable, int offset = 0) : m_variable(variable), m_offset(offset) {}

	// Properties

	const typename S::template VariableType<T> *GetVariable() const { return m_variable; }
	typename S::template VariableType<T> *GetVariable() { return m_variable; }
	void SetVariable(typename S::template VariableType<T> *variable) { m_variable = variable; }

	int GetOffset() const { return m_offset; }
	void SetOffset(int offset) { m_offset = offset; }

	MemoryAddress<B, T, S> *CreateOffsetAddress(int offset) const override
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

private:
	typename S::template VariableType<T> *m_variable = nullptr;
	int m_offset = 0;
};

template<class T, class S = AddressableSpace>
using MemoryAddress32 = MemoryAddress<Bits::Bits32, T, S>;
template<class T, class S = AddressableSpace>
using MemoryAddress64 = MemoryAddress<Bits::Bits64, T, S>;

}
