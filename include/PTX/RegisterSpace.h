#pragma once

#include <string>
#include <sstream>

#include "PTX/StateSpace.h"

namespace PTX {

template<typename T>
class RegisterSpace : public StateSpace<T>
{
public:
	RegisterSpace(std::string prefix, unsigned int count) : m_prefix(prefix), m_count(count) {}
	RegisterSpace(std::vector<std::string> names) : m_names(names) {}

	std::string SpaceName() { return ".reg"; }
	std::string GetName() { return "<uerror>"; }

	std::string ToString()
	{
		std::ostringstream code;
		code << ".reg " + TypeName<T>() + " ";
		if (m_count > 0)
		{
			code << m_prefix + std::string("<") + std::to_string(m_count) + std::string(">");
		}
		else
		{
			bool first = true;
			for (std::vector<std::string>::iterator it = m_names.begin(); it != m_names.end(); ++it)
			{
				if (!first)
				{
					code << ", ";
					first = false;
				}
				code << *it;
			}
		}
		code << ";" << std::endl;
		return code.str();
	}

private:
	std::string m_prefix = "";
	unsigned int m_count = 0;

	std::vector<std::string> m_names;
};

}
