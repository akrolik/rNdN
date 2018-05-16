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
		if (m_alignment != 4)
		{
			return ".ptr.align " + std::to_string(m_alignment);
		}
		return "";
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

	//TODO: space
	std::string Specifier() const { return ".param"; }

protected:
	using StateSpace<T>::m_names;

	unsigned int m_alignment = 4;
};

template<class T, AddressSpace A = Generic>
class ParameterSpace : public AddressableSpace<T, Param>
{
public:
	using AddressableSpace<T, Param>::AddressableSpace;

	std::string Directives() const
	{
		std::ostringstream code;
		if (A != Generic || m_alignment != 4)
		{
			code << ".ptr";
			if (A != Generic)
			{
				code << A;
			}
			if (m_alignment != 4)
			{
				code << ".align " << m_alignment;
			}
			code << " ";
		}
		return code.str();
	}

protected:
	using AddressableSpace<T, Param>::m_alignment;
};

}
