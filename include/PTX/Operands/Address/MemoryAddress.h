#pragma once

#include "PTX/Operands/Address/Address.h"

#include "PTX/Operands/Variables/AddressableVariable.h"

namespace PTX {

template<Bits A, class T, AddressSpace S>
class MemoryAddress : public Address<A, T, S>
{
public:
	MemoryAddress(AddressableVariable<T, S> *variable, int offset = 0) : m_variable(variable), m_offset(offset) {}

	std::string ToString() const
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

	AddressableVariable<T, S> *GetVariable() const { return m_variable; }
	int GetOffset() const { return m_offset; }

private:
	AddressableVariable<T, S> *m_variable = nullptr;
	int m_offset = 0;
};

template<class T, AddressSpace S>
using MemoryAddress32 = MemoryAddress<Bits::Bits32, T, S>;
template<class T, AddressSpace S>
using MemoryAddress64 = MemoryAddress<Bits::Bits64, T, S>;

}
