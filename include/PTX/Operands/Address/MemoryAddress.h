#pragma once

#include "PTX/Operands/Address/Address.h"

namespace PTX {

template<Bits B, class T, class S = AddressableSpace>
class MemoryAddress : public Address<B, T, S>
{
public:
	REQUIRE_TYPE_PARAM(MemoryAddress,
		REQUIRE_BASE(T, DataType)
	);
	REQUIRE_SPACE_PARAM(MemoryAddress,
		REQUIRE_BASE(S, AddressableSpace)
	);

	MemoryAddress(const typename S::template VariableType<T> *variable, int offset = 0) : m_variable(variable), m_offset(offset) {}

	std::string ToString() const override
	{
		if (m_offset > 0)
		{
			return m_variable->ToString() + "+" + std::to_string(m_offset);
		}
		else if (m_offset < 0)
		{
			return m_variable->ToString() + std::to_string(m_offset);
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

	const typename S::template VariableType<T> *GetVariable() const { return m_variable; }
	int GetOffset() const { return m_offset; }

private:
	const typename S::template VariableType<T> *m_variable = nullptr;
	int m_offset = 0;
};

template<class T, class S = AddressableSpace>
using MemoryAddress32 = MemoryAddress<Bits::Bits32, T, S>;
template<class T, class S = AddressableSpace>
using MemoryAddress64 = MemoryAddress<Bits::Bits64, T, S>;

}
