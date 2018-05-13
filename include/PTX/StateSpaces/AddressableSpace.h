#pragma once

#include "PTX/StateSpaces/StateSpace.h"

#include "PTX/Operands/Variables/Variable.h"

namespace PTX {

template<class T, AddressSpace A>
class AddressableSpace : public StateSpace<T>
{
	static_assert(std::is_base_of<ValueType, T>::value, "T must be a PTX::ValueType");
public:
	using StateSpace<T>::StateSpace;

	void SetAlignment(unsigned int alignment) { m_alignment = alignment; }
	unsigned int GetAlignment() const { return m_alignment; }

	std::string Directives() const
	{
		std::ostringstream code;
		if (m_addressSpace != A || m_alignment != 4)
		{
			code << ".ptr";
			if (m_addressSpace != A)
			{
				code << GetAddressSpaceName(m_addressSpace);
			}
			if (m_alignment != 4)
			{
				code << ".align " << m_alignment;
			}
			code << " ";
		}
		return code.str();
	}

	virtual Variable<T, AddressableSpace<T, A>> *GetVariable(std::string name, unsigned int index = 0)
	{
		for (typename std::vector<NameSet>::const_iterator it = m_names.begin(); it != m_names.end(); ++it)
		{
			if (it->GetPrefix() == name)
			{
				return new Variable<T, AddressableSpace<T, A>>(it->GetName(index), this);
			}
		}
		std::cerr << "[Error] Variable " << name << " not found in StateSpace" << std::endl;
		std::exit(EXIT_FAILURE);
	}

	
	void SetAddressSpace(std::enable_if_t<A == Param, AddressSpace> addressSpace) { m_addressSpace = addressSpace; }
	// AddressSpace GetAddressSpace(std::enable_if_t<A == Param, void>) const { return m_addressSpace; }
	// AddressSpace GetAddressSpace(std::enable_if_t<A != Param, void>) const { return m_addressSpace; }

	std::string Specifier() const { return ".param"; }

protected:
	using StateSpace<T>::m_names;

	unsigned int m_alignment = 4;
	AddressSpace m_addressSpace = A;
};


template<class T>
using ParameterSpace = AddressableSpace<T, Param>;

}
