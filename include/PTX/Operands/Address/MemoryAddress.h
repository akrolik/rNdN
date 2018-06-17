#pragma once

#include "PTX/Operands/Address/Address.h"

namespace PTX {

template<Bits B, class T, class S = AddressableSpace>
class MemoryAddress : public Address<B, T, S>
{
	REQUIRE_BASE_TYPE(MemoryAddress, DataType);
	REQUIRE_BASE_SPACE(MemoryAddress, AddressableSpace);
public:
	MemoryAddress(const typename S::template VariableType<T> *variable, int offset = 0) : m_variable(variable), m_offset(offset) {}

	std::string ToString() const override
	{
		if (m_offset > 0)
		{
			return "[" + m_variable->ToString() + "+" + std::to_string(m_offset) + "]";
		}
		else if (m_offset < 0)
		{
			return "[" + m_variable->ToString() + std::to_string(m_offset) + "]";
		}
		else
		{
			return "[" + m_variable->ToString() + "]";
		}
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
