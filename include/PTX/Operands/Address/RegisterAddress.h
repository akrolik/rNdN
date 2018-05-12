#pragma once

#include "PTX/Operands/Address/Address.h"
#include "PTX/Operands/Variable.h"

namespace PTX {

template<Bits A, class T>
class RegisterAddress : public Address<A, T>
{
	static_assert(std::is_base_of<Type, T>::value, "T must be a PTX::Type");
public:
	RegisterAddress(AddressRegister<A, T> *reg, int offset = 0) : m_register(reg), m_offset(offset) {}

	AddressSpace GetSpace() const { return m_register->GetAddressSpace(); }

	std::string ToString() const
	{
		if (m_offset > 0)
		{
			return "[" + m_register->ToString() + "+" + std::to_string(m_offset) + "]";
		}
		else if (m_offset < 0)
		{
			return "[" + m_register->ToString() + std::to_string(m_offset) + "]";
		}
		else
		{
			return "[" + m_register->ToString() + "]";
		}
	}

	AddressRegister<A, T> *GetRegister() const { return m_register; }
	int GetOffset() const { return m_offset; }

private:
	AddressRegister<A, T> *m_register = nullptr;
	int m_offset = 0;
};

template<class T>
using RegisterAddress32 = RegisterAddress<Bits::Bits32, T>;
template<class T>
using RegisterAddress64 = RegisterAddress<Bits::Bits64, T>;

}
