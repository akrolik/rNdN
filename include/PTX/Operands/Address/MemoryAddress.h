#pragma once

#include "PTX/Operands/Address/Address.h"

#include "PTX/Operands/Variables/AddressableVariable.h"

namespace PTX {

template<Bits B, class T, AddressSpace A>
class MemoryAddress : public Address<B, T, A>
{
public:
	MemoryAddress(AddressableVariable<T, A> *variable, int offset = 0) : m_variable(variable), m_offset(offset) {}

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

	AddressableVariable<T, A> *GetVariable() const { return m_variable; }
	int GetOffset() const { return m_offset; }

private:
	AddressableVariable<T, A> *m_variable = nullptr;
	int m_offset = 0;
};

template<class T, AddressSpace A>
using MemoryAddress32 = MemoryAddress<Bits::Bits32, T, A>;
template<class T, AddressSpace A>
using MemoryAddress64 = MemoryAddress<Bits::Bits64, T, A>;

}
