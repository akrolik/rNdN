#pragma once

#include "PTX/Operands/Address/Address.h"
#include "PTX/Operands/Register/AddressRegister.h"

namespace PTX {

template<Bits A, class T, VectorSize V = Scalar>
class RegisterAddress : public Address<A, T, V>
{
	static_assert(std::is_base_of<Type, T>::value, "T must be a PTX::Type");
public:
	RegisterAddress(AddressRegister<A> *reg, int offset = 0) : m_register(reg), m_offset(offset) {}

	AddressSpace GetSpace() const { return m_register->GetAddressSpace(); }

	std::string ToString() const
	{
		if (m_offset > 0)
		{
			return "[" + m_register->GetName() + "+" + std::to_string(m_offset) + "]";
		}
		else if (m_offset < 0)
		{
			return "[" + m_register->GetName() + std::to_string(m_offset) + "]";
		}
		else
		{
			return "[" + m_register->GetName() + "]";
		}
	}

	AddressRegister<A> *GetRegister() const { return m_register; }
	int GetOffset() const { return m_offset; }

private:
	AddressRegister<A> *m_register = nullptr;
	int m_offset = 0;
};

}
