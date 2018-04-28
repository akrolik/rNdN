#pragma once

#include <iostream>
#include <sstream>
#include <string>
#include <vector>

#include "PTX/StateSpace.h"

namespace PTX {

template<class T, VectorSize V>
class Register;

template<class T, VectorSize V>
class IndexedRegister;

template<class T, VectorSize V = Scalar>
class RegisterSpace : public StateSpace<T, V>
{
	static_assert(std::is_base_of<Type, T>::value, "T must be a PTX::Type");
public:
	struct Element {
	public:
		Element(std::string name, unsigned int count = 1) : m_name(name), m_count(count) {}

		std::string VariableName(unsigned int index)
		{
			if (index >= m_count)
			{
				std::cerr << "[Error] Register out of bounds" << std::endl;
				std::exit(EXIT_FAILURE);
			}

			if (m_count > 1)
			{
				return m_name + std::to_string(index + 1);
			}
			return m_name;
		}

		std::string Name(unsigned int index)
		{
			return VariableName(index);
		}

		std::string ToString()
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

	RegisterSpace(std::string prefix, unsigned int count)
	{
		m_elements.push_back(new Element(prefix, count));
	}

	RegisterSpace(std::string name)
	{
		m_elements.push_back(new Element(name));
	}

	RegisterSpace(std::vector<std::string> names)
	{
		for (typename std::vector<std::string>::const_iterator it = names.begin(); it != names.end(); ++it)
		{
			m_elements.push_back(new Element(*it));
		}
	}

	Register<T, V> *GetRegister(std::string& name);
	Register<T, V> *GetRegister(unsigned int index, unsigned int element = 0);

	IndexedRegister<T, V> *GetRegister(unsigned int index, VectorElement vectorElement, unsigned int element = 0);
	IndexedRegister<T, V> *GetRegister(std::string& name, VectorElement vectorElement, unsigned int element = 0);

	std::string SpaceName() { return ".reg"; }

	std::string ElementNames()
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

	std::string ToString()
	{
		return "\t" + SpaceName() + " " + TypeName<T>() + " " + ElementNames() + ";\n";
	}

private:
	std::vector<Element *> m_elements;
};

#include "PTX/Operands/Register.h"
// #include "PTX/Operands/IndexedRegister.h"

template<class T, VectorSize V>
Register<T, V> *RegisterSpace<T, V>::GetRegister(unsigned int index, unsigned int element)
{
	return new Register<T, V>(m_elements.at(element), index);
}

template<class T, VectorSize V>
Register<T, V> *RegisterSpace<T, V>::GetRegister(std::string& name)
{
	for (typename std::vector<Element *>::const_iterator it = m_elements.begin(); it != m_elements.end(); ++it)
	{
		if ((*it)->VariableName(0) == name)
		{
			return new Register<T, V>(*it, 0);
		}
	}
	return nullptr;
}

template<class T, VectorSize V>
IndexedRegister<T, V> *RegisterSpace<T, V>::GetRegister(unsigned int index, VectorElement vectorElement, unsigned int element)
{
	return new IndexedRegister<T, V>(m_elements.at(element), index, vectorElement);
}

template<class T, VectorSize V>
IndexedRegister<T, V> *RegisterSpace<T, V>::GetRegister(std::string& name, VectorElement vectorElement, unsigned int element)
{
	for (typename std::vector<Element *>::const_iterator it = m_elements.begin(); it != m_elements.end(); ++it)
	{
		if ((*it)->VariableName(0) == name)
		{
			return new IndexedRegister<T, V>(*it, 0, vectorElement);
		}
	}
	return nullptr;
}

}
