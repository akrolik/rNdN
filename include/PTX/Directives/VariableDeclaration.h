#pragma once

#include <string>
#include <sstream>

#include "PTX/Statements/DirectiveStatement.h"

#include "PTX/Type.h"
#include "PTX/StateSpace.h"

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

template<class T, class S>
class VariableDeclaration : public DirectiveStatement
{
	REQUIRE_BASE_TYPE(VariableDeclaration, Type);
	REQUIRE_BASE_SPACE(VariableDeclaration, StateSpace);
public:
	virtual std::string Directives() const { return ""; }

	VariableDeclaration(std::string prefix, unsigned int count = 1)
	{
		m_names.emplace_back(prefix, count);
	}

	VariableDeclaration(std::vector<std::string> names)
	{
		for (std::vector<std::string>::const_iterator it = names.begin(); it != names.end(); ++it)
		{
			m_names.emplace_back(*it);
		}
	}

	VariableDeclaration(std::vector<NameSet> variables) : m_names(variables) {}

	typename S::template VariableType<T> *GetVariable(std::string name, unsigned int index = 0)
	{
		for (typename std::vector<NameSet>::const_iterator it = m_names.begin(); it != m_names.end(); ++it)
		{
			if (it->GetPrefix() == name)
			{
				return new typename S::template VariableType<T>(it->GetName(index));
			}
		}
		std::cerr << "[Error] PTX::Variable(" << name << ") not found in PTX::VariableDeclaration" << std::endl;
		std::exit(EXIT_FAILURE);
	}

	std::vector<NameSet> GetNames() const { return m_names; }

	std::string ToString() const
	{
		return S::Name() + " " + T::Name() + " " + Directives() + VariableNames();
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

template<class T>
using RegisterDeclaration = VariableDeclaration<T, RegisterSpace>;

template<class T>
using SpecialRegisterDeclaration = VariableDeclaration<T, SpecialRegisterSpace>;

template<class T>
using ParameterDeclaration = VariableDeclaration<T, ParameterSpace>;

template<class T, Bits B, class S = AddressableSpace>
class PointerDeclaration : public ParameterDeclaration<PointerType<T, B, S>>
{
public:
	using VariableDeclaration<PointerType<T, B, S>, ParameterSpace>::VariableDeclaration;

	void SetAlignment(unsigned int alignment) { m_alignment = alignment; }

	unsigned int GetAlignment() const { return m_alignment; }

	std::string Directives() const
	{
		std::ostringstream code;
		if (!std::is_same<S, AddressableSpace>::value || m_alignment != 4)
		{
			code << ".ptr";
			if (!std::is_same<S, AddressableSpace>::value)
			{
				code << S::Name();
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
	unsigned int m_alignment = 4;
};

template<class T, class S = AddressableSpace>
using Pointer32Declaration = PointerDeclaration<T, Bits::Bits32, S>;
template<class T, class S = AddressableSpace>
using Pointer64Declaration = PointerDeclaration<T, Bits::Bits64, S>;
}
