#pragma once

#include "PTX/Operands/Address/Address.h"

#include "PTX/Operands/Variables/Register.h"

namespace PTX {

template<Bits A, class T, AddressSpace S>
class RegisterAddress : public Address<A, T, S>
{
public:
	RegisterAddress(Register<UIntType<A>> *reg, int offset = 0) : m_variable(reg), m_offset(offset) {}

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

	Register<UIntType<A>> *GetRegister() const { return m_variable; }
	int GetOffset() const { return m_offset; }

private:
	Register<UIntType<A>> *m_variable;
	int m_offset = 0;
};

template<class T, AddressSpace S>
using RegisterAddress32 = RegisterAddress<Bits::Bits32, T, S>;
template<class T, AddressSpace S>
using RegisterAddress64 = RegisterAddress<Bits::Bits64, T, S>;

}
