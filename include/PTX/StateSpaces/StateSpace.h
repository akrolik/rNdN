#pragma once

#include "PTX/Statements/DirectiveStatement.h"
#include "PTX/Type.h"

namespace PTX {

struct Structure
{
public:
	Structure(std::string name, unsigned int count = 1) : m_name(name), m_count(count) {}

	std::string GetName() const { return m_name; }

	std::string GetName(unsigned int index) const
	{
		if (index >= m_count)
		{
			std::cerr << "[Error] StateSpace element out of bounds" << std::endl;
			std::exit(EXIT_FAILURE);
		}

		if (m_count > 1)
		{
			return m_name + std::to_string(index + 1);
		}
		return m_name;
	}

	std::string ToString() const
	{
		if (m_count > 1)
		{
			return m_name + "<" + std::to_string(m_count + 1) + ">";
		}

		return m_name;
	}
private:
	std::string m_name;
	unsigned int m_count;
};

template<class T, VectorSize V = Scalar>
class StateSpace : public DirectiveStatement
{
	static_assert(std::is_base_of<Type, T>::value, "T must be a PTX::Type");
public:
	struct Element : public Structure
	{
		using Structure::Structure;
	};

	StateSpace(std::string prefix, unsigned int count)
	{
		m_elements.push_back(new Element(prefix, count));
	}

	StateSpace(std::string name)
	{
		m_elements.push_back(new Element(name));
	}

	StateSpace(std::vector<std::string> names)
	{
		for (typename std::vector<std::string>::const_iterator it = names.begin(); it != names.end(); ++it)
		{
			m_elements.push_back(new Element(*it));
		}
	}

	virtual std::string Specifier() const = 0;

	std::string GetElementNames() const
	{
		std::ostringstream code;
		bool first = true;
		for (typename std::vector<Element *>::const_iterator it = m_elements.begin(); it != m_elements.end(); ++it)
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

	std::string ToString() const
	{
		return "\t" + Specifier() + " " + TypeName<T>() + " " + GetElementNames() + ";\n";
	}

protected:
	std::vector<Element *> m_elements;

	Structure *GetStructure(std::string name) const
	{
		for (typename std::vector<typename StateSpace<T, V>::Element *>::const_iterator it = m_elements.begin(); it != m_elements.end(); ++it)
		{
			if ((*it)->GetName() == name)
			{
				return *it;
			}
		}
		std::cerr << "[Error] StateSpace element '" << name << "' not found" << std::endl;
		std::exit(EXIT_FAILURE);
	}
};

}
