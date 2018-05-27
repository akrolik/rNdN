#pragma once

#include "PTX/Statements/DirectiveStatement.h"

#include "PTX/Type.h"

namespace PTX {

class NameSet
{
public:
	NameSet(std::string name, unsigned int count = 1) : m_name(name), m_count(count) {}

	virtual std::string GetPrefix() const
	{
		return m_name;
	}

	virtual std::string GetName(unsigned int index) const
	{
		if (index >= m_count)
		{
			std::cerr << "[Error] Variable index " << index << " out of bounds" << std::endl;
			std::exit(EXIT_FAILURE);
		}

		if (m_count > 1)
		{
			return m_name + std::to_string(index);

		}
		return m_name;
	}
	
	virtual std::string ToString() const
	{
		if (m_count > 1)
		{
			return m_name + "<" + std::to_string(m_count) + ">";
		}
		return m_name;
	}

protected:
	std::string m_name;
	unsigned int m_count;
};

template<class T>
class StateSpace : public DirectiveStatement
{
	REQUIRE_BASE_TYPE(StateSpace, Type);
public:
	using SpaceType = T;

	virtual std::string Specifier() const = 0;
	virtual std::string Directives() const { return ""; }

	StateSpace(std::string prefix, unsigned int count = 1)
	{
		m_names.emplace_back(prefix, count);
	}

	StateSpace(std::vector<std::string> names)
	{
		for (std::vector<std::string>::const_iterator it = names.begin(); it != names.end(); ++it)
		{
			m_names.emplace_back(*it);
		}
	}

	StateSpace(std::vector<NameSet> variables) : m_names(variables) {}

	std::vector<NameSet> GetNames() const { return m_names; }

	std::string ToString() const
	{
		return Specifier() + " " + T::Name() + " " + Directives() + VariableNames();
	}

protected:
	virtual std::string VariableNames() const
	{
		std::ostringstream code;
		bool first = true;
		for (typename std::vector<NameSet>::const_iterator it = m_names.begin(); it != m_names.end(); ++it)
		{
			if (!first)
			{
				code << ", ";
				first = false;
			}
			code << it->ToString();
		}
		return code.str();
	}

	std::vector<NameSet> m_names;
};

}
