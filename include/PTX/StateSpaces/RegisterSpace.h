#pragma once

#include "PTX/StateSpaces/StateSpace.h"

#include "PTX/Operands/Variables/Variable.h"

namespace PTX {

template<class T>
class RegisterSpace : public StateSpace<T>
{
	REQUIRE_BASE_TYPE(RegisterSpace, ValueType);
public:
	using StateSpace<T>::StateSpace;

	virtual Variable<T, RegisterSpace<T>> *GetVariable(std::string name, unsigned int index = 0)
	{
		for (typename std::vector<NameSet>::const_iterator it = m_names.begin(); it != m_names.end(); ++it)
		{
			if (it->GetPrefix() == name)
			{
				return new Variable<T, RegisterSpace<T>>(it->GetName(index), this);
			}
		}
		std::cerr << "[Error] PTX::Variable(" << name << ") not found in PTX::RegisterSpace" << std::endl;
		std::exit(EXIT_FAILURE);
	}

	std::string Specifier() const { return ".reg"; }

protected:
	using StateSpace<T>::m_names;
};

}
