#pragma once

#include "PTX/StateSpaces/StateSpace.h"

#include "PTX/Operands/Variables/Variable.h"

namespace PTX {

template<class T, AddressSpace A>
class AddressableSpace : public StateSpace<T>
{
	REQUIRE_TYPE(AddressableSpace, DataType);
public:
	using StateSpace<T>::StateSpace;

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

protected:
	using StateSpace<T>::m_names;
};

}
