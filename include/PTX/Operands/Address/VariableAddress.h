#pragma once

#include "PTX/Operands/Address/Address.h"
#include "PTX/Operands/Variable.h"

namespace PTX {

template<Bits A, class T, VectorSize V = Scalar>
class VariableAddress : public Address<A, T, V>
{
	static_assert(std::is_base_of<Type, T>::value, "T must be a PTX::Type");
public:
	VariableAddress(Variable<T, V> *variable, int offset = 0) : m_variable(variable), m_offset(offset) {}

	AddressSpace GetSpace() const { return m_variable->GetAddressSpace(); }

	std::string ToString() const
	{
		if (m_offset > 0)
		{
			return "[" + m_variable->GetName() + "+" + std::to_string(m_offset) + "]";
		}
		else if (m_offset < 0)
		{
			return "[" + m_variable->GetName() + std::to_string(m_offset) + "]";
		}
		else
		{
			return "[" + m_variable->GetName() + "]";
		}
	}

	Variable<T, V> *GetVariable() const { return m_variable; }
	int GetOffset() const { return m_offset; }

private:
	Variable<T, V> *m_variable = nullptr;
	int m_offset = 0;
};

}
