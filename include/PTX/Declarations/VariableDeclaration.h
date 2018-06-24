#pragma once

#include <string>
#include <sstream>
#include <iostream>
#include <vector>

#include "PTX/Declarations/Declaration.h"
#include "PTX/Statements/DirectiveStatement.h"

#include "PTX/Type.h"
#include "PTX/StateSpace.h"

namespace PTX {

class NameSet
{
public:
	NameSet(const std::string& name, unsigned int count = 1) : m_name(name), m_count(count) {}

	virtual std::string GetPrefix() const
	{
		return m_name;
	}

	virtual std::string GetName(unsigned int index) const
	{
		if (index >= m_count)
		{
			std::cerr << "[ERROR] Variable index " << index << " out of bounds" << std::endl;
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

template<class S>
class UntypedVariableDeclaration : public DirectiveStatement, public Declaration
{

};

template<class T, class S>
class VariableDeclaration : public UntypedVariableDeclaration<S>
{
	REQUIRE_BASE_TYPE(VariableDeclaration, Type);
	REQUIRE_BASE_SPACE(VariableDeclaration, StateSpace);
public:
	VariableDeclaration() {}

	VariableDeclaration(const std::string& prefix, unsigned int count = 1)
	{
		AddNames(prefix, count);
	}

	VariableDeclaration(const std::vector<std::string>& names)
	{
		AddNames(names);
	}

	VariableDeclaration(const std::vector<NameSet>& variables) : m_names(variables) {}

	virtual std::string Directives() const { return ""; }

	void AddNames(const std::string& prefix, unsigned int count = 1)
	{
		m_names.emplace_back(prefix, count);
	}

	void AddNames(const std::vector<std::string>& names)
	{
		for (const auto& name : names)
		{
			m_names.emplace_back(name);
		}
	}

	const typename S::template VariableType<T> *GetVariable(const std::string& name, unsigned int index = 0)
	{
		for (const auto &set : m_names)
		{
			if (set.GetPrefix() == name)
			{
				return new typename S::template VariableType<T>(set.GetName(index));
			}
		}
		std::cerr << "[ERROR] PTX::Variable(" << name << ") not found in PTX::VariableDeclaration" << std::endl;
		std::exit(EXIT_FAILURE);
	}

	const std::vector<NameSet>& GetNames() const { return m_names; }

	std::string ToString() const override
	{
		return S::Name() + " " + T::Name() + " " + Directives() + VariableNames();
	}

	json ToJSON() const override
	{
		json j;
		j["kind"] = "PTX::VariableDeclaration";
		j["type"] = T::Name();
		j["space"] = S::Name();
		std::string directives = Directives();
		if (directives.length() > 0)
		{
			j["directives"] = directives;
		}
		if (m_names.size() == 1)
		{
			j["names"] = m_names.at(0).ToString();
		}
		else
		{
			for (const auto& set : m_names)
			{
				j["names"].push_back(set.ToString());
			}
		}
		return j;
	}

protected:
	virtual std::string VariableNames() const
	{
		std::ostringstream code;
		bool first = true;
		for (const auto& set : m_names)
		{
			if (!first)
			{
				code << ", ";
			}
			first = false;
			code << set.ToString();
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

	std::string Directives() const override
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
