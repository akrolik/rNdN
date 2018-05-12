#pragma once

#include "PTX/StateSpaces/StateSpace.h"

namespace PTX {

template<class T>
class RegisterSpace : public StateSpace<T>
{
public:
	RegisterSpace() {}

	RegisterSpace(std::string prefix, unsigned int count)
	{
		m_variables.push_back(new VariableSet<T, RegisterSpace<T>>(prefix, count, this));
	}

	RegisterSpace(std::string name)
	{
		m_variables.push_back(new Variable<T, RegisterSpace<T>>(name, this));
	}

	RegisterSpace(std::vector<std::string> names)
	{
		for (std::vector<std::string>::const_iterator it = names.begin(); it != names.end(); ++it)
		{
			m_variables.push_back(new Variable<T, RegisterSpace<T>>(*it, this));
		}
	}

	virtual Variable<T, RegisterSpace<T>> *GetVariable(std::string name, unsigned int index = 0) const
	{
		for (typename std::vector<VariableSet<T, RegisterSpace<T>> *>::const_iterator it = m_variables.begin(); it != m_variables.end(); ++it)
		{
			if ((*it)->GetPrefix() == name)
			{
				return (*it)->GetVariable(index);
			}
		}
		std::cerr << "[Error] Variable " << name << " not found in StateSpace" << std::endl;
		std::exit(EXIT_FAILURE);
	}

	std::string Specifier() const { return ".reg"; }

protected:
	std::string VariableNames() const
	{
		std::ostringstream code;
		bool first = true;
		for (typename std::vector<VariableSet<T, RegisterSpace<T>> *>::const_iterator it = m_variables.begin(); it != m_variables.end(); ++it)
		{
			if (!first)
			{
				code << ", ";
				first = false;
			}
			code << (*it)->ToString();
		}
		return code.str();
	}

	std::vector<VariableSet<T, RegisterSpace<T>> *> m_variables;
};

}
