#pragma once

#include "PTX/Operands/Address/Address.h"

#include "PTX/StateSpace.h"
#include "PTX/Operands/Variables/Register.h"

namespace PTX {

template<Bits B, class T, class S>
class RegisterAddress : public Address<B, T, S>
{
	REQUIRE_BASE_TYPE(RegisterAddress, Type);
	REQUIRE_BASE_SPACE(RegisterAddress, AddressableSpace);
public:
	RegisterAddress(Register<PointerType<T, B, S>> *variable, int offset = 0) : m_variable(variable), m_offset(offset) {}

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

	Register<PointerType<T, B, S>> *GetRegister() const { return m_variable; }
	int GetOffset() const { return m_offset; }

private:
	Register<PointerType<T, B, S>> *m_variable;
	int m_offset = 0;
};

template<class T, class S>
using RegisterAddress32 = RegisterAddress<Bits::Bits32, T, S>;
template<class T, class S>
using RegisterAddress64 = RegisterAddress<Bits::Bits64, T, S>;

}
