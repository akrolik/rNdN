#pragma once

#include "PTX/Operands/Address/Address.h"
#include "PTX/Operands/Variable.h"

namespace PTX {

template<Bits A, class T>
class MemoryAddress : public Address<A, T>
{
	static_assert(std::is_base_of<Type, T>::value, "T must be a PTX::Type");
public:
	MemoryAddress(MemoryVariable<T> *variable, int offset = 0) : m_variable(variable), m_offset(offset) {}

	AddressSpace GetSpace() const { return m_variable->GetStateSpace()->GetAddressSpace(); }

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

	MemoryVariable<T> *GetVariable() const { return m_variable; }
	int GetOffset() const { return m_offset; }

private:
	MemoryVariable<T> *m_variable = nullptr;
	int m_offset = 0;
};

template<class T>
using MemoryAddress32 = MemoryAddress<Bits::Bits32, T>;
template<class T>
using MemoryAddress64 = MemoryAddress<Bits::Bits64, T>;

}
