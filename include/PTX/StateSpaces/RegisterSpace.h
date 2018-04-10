#pragma once

#include <sstream>

#include "PTX/StateSpace.h"

namespace PTX {

template<Type T, VectorSize V = Scalar>
class RegisterSpace : public StateSpace<T, V>
{
public:
	RegisterSpace(std::string prefix, unsigned int count) : m_prefix(prefix), m_count(count) {}
	RegisterSpace(std::vector<std::string> names) : m_names(names) {}

	std::string SpaceName() { return ".reg"; }
	std::string Name()
	{
		std::ostringstream code;
		if (m_count > 0)
		{
			code << m_prefix + std::string("<") + std::to_string(m_count + 1) + std::string(">");
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
		return code.str();
	}

	std::string Name(unsigned int index)
	{
		if (m_count > 0)
		{
			if (index >= m_count)
			{
				std::cerr << "[Error] register out of bounds" << std::endl;
				std::exit(EXIT_FAILURE);
			}
			return m_prefix + std::to_string(index + 1);
		}

		return m_names.at(index);
	}

	std::string ToString()
	{
		return "\t" + SpaceName() + " " + TypeName<T>() + " " + Name() + "\n";
	}

private:
	std::string m_prefix = "";
	unsigned int m_count = 0;

	std::vector<std::string> m_names;
};

}
