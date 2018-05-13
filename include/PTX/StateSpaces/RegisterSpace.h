#pragma once

#include "PTX/StateSpaces/StateSpace.h"

#include "PTX/Operands/Variable.h"

namespace PTX {

template<class T>
class RegisterSpace : public StateSpace<T>
{
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
		std::cerr << "[Error] Variable " << name << " not found in StateSpace" << std::endl;
		std::exit(EXIT_FAILURE);
	}

	std::string Specifier() const { return ".reg"; }

protected:
	using StateSpace<T>::m_names;
};

template<class T>
using Register = Variable<T, RegisterSpace<T>>;

}
