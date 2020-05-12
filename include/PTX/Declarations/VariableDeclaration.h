#pragma once

#include <string>
#include <sstream>
#include <iostream>
#include <vector>

#include "PTX/Declarations/Declaration.h"
#include "PTX/Statements/DirectiveStatement.h"

#include "PTX/Type.h"
#include "PTX/StateSpace.h"

#include "Utils/Logger.h"

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

	void SetCount(unsigned int count) { m_count = count; }
	
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

class VariableDeclaration : public DirectiveStatement, public Declaration
{
public:
	using Declaration::Declaration;

	VariableDeclaration() {}

	VariableDeclaration(const std::string& prefix, unsigned int count = 1)
	{
		AddNames(prefix, count);
	}

	VariableDeclaration(const std::vector<std::string>& names)
	{
		AddNames(names);
	}

	VariableDeclaration(const std::vector<NameSet *>& variables) : m_names(variables) {}

	virtual std::string PreDirectives() const { return ""; }
	virtual std::string PostDirectives() const { return ""; }

	void AddNames(const std::string& prefix, unsigned int count = 1)
	{
		m_names.push_back(new NameSet(prefix, count));
	}

	void AddNames(const std::vector<std::string>& names)
	{
		for (const auto& name : names)
		{
			m_names.push_back(new NameSet(name));
		}
	}

	void UpdateName(const std::string& prefix, unsigned int count)
	{
		for (auto& set : m_names)
		{
			if (set->GetPrefix() == prefix)
			{
				set->SetCount(count);
				return;
			}
		}

		AddNames(prefix, count);
	}

	const std::vector<NameSet *>& GetNames() const { return m_names; }

	virtual std::string ToString(unsigned int indentation) const = 0;
	virtual std::string ToString(unsigned int indentation, bool terminate) const = 0;

	json ToJSON() const override
	{
		json j;
		j["kind"] = "PTX::VariableDeclaration";
		j["type"] = "<untyped>";
		j["space"] = "<unspaced>";
		std::string preDirectives = PreDirectives();
		if (preDirectives.length() > 0)
		{
			j["pre_directives"] = preDirectives;
		}
		std::string postDirectives = PostDirectives();
		if (postDirectives.length() > 0)
		{
			j["post_directives"] = postDirectives;
		}
		if (m_names.size() == 1)
		{
			j["names"] = m_names.at(0)->ToString();
		}
		else
		{
			for (const auto& set : m_names)
			{
				j["names"].push_back(set->ToString());
			}
		}
		return j;
	}

protected:
	std::vector<NameSet *> m_names;
};

template<class T, class S, typename Enabled = void>
class TypedVariableDeclarationBase : public VariableDeclaration
{
public:
	using VariableDeclaration::VariableDeclaration;
};

template<class T, class S>
class TypedVariableDeclarationBase<T, S, std::enable_if_t<REQUIRE_BASE(S, AddressableSpace)>> : public VariableDeclaration
{
public:
	using VariableDeclaration::VariableDeclaration;

	const unsigned int DefaultAlignment = BitSize<T::TypeBits>::NumBytes;

	void SetAlignment(unsigned int alignment) { m_alignment = alignment; }
	unsigned int GetAlignment() const { return m_alignment; }

	std::string PreDirectives() const override
	{
		if (m_alignment != DefaultAlignment)
		{
			return ".align " + std::to_string(m_alignment) + " ";
		}
		return "";
	}

protected:
	unsigned int m_alignment = DefaultAlignment;
};

template<class T, class S>
class TypedVariableDeclaration : public TypedVariableDeclarationBase<T, S>
{
public:
	REQUIRE_TYPE_PARAM(VariableDeclaration,
		REQUIRE_BASE(T, Type)
	);

	REQUIRE_SPACE_PARAM(VariableDeclaration,
		REQUIRE_BASE(S, StateSpace)
	);

	using TypedVariableDeclarationBase<T, S>::TypedVariableDeclarationBase;

	const typename S::template VariableType<T> *GetVariable(const std::string& name, unsigned int index = 0) const
	{
		for (const auto &set : this->m_names)
		{
			if (set->GetPrefix() == name)
			{
				return new typename S::template VariableType<T>(set, index);
			}
		}
		Utils::Logger::LogError("PTX::Variable(" + name + ") not found in PTX::VariableDeclaration");
	}

	std::string ToString(unsigned int indentation) const override
	{
		return ToString(indentation, true);
	}

	std::string ToString(unsigned int indentation, bool terminate) const override
	{
		std::string code = std::string(indentation, '\t');
		if (this->m_linkDirective != Declaration::LinkDirective::None)
		{
			code += this->LinkDirectiveString(this->m_linkDirective) + " ";
		}
		code += S::Name() + " " + this->PreDirectives();
		if constexpr(is_array_type<T>::value)
		{
			code += T::BaseName();
		}
		else
		{
			code += T::Name();
		}
		code += " " + this->PostDirectives() + VariableNames();
		if (terminate)
		{
			code += ";";
		}
		return code;
	}

	json ToJSON() const override
	{
		json j = VariableDeclaration::ToJSON();
		j["type"] = T::Name();
		j["space"] = S::Name();
		return j;
	}

protected:
	std::string VariableNames() const
	{
		std::string code;
		bool first = true;
		for (const auto& set : this->m_names)
		{
			if (!first)
			{
				code += ", ";
			}
			first = false;
			code += set->ToString();
			if constexpr(is_array_type<T>::value)
			{
				code += T::Dimensions();
			}
		}
		return code;
	}
};

template<class T, class S>
class InitializedVariableDeclaration : public TypedVariableDeclaration<T, S>
{
public:
	REQUIRE_TYPE_PARAM(VariableDeclaration,
		REQUIRE_BASE(T, Type)
	);

	REQUIRE_SPACE_PARAM(VariableDeclaration,
		REQUIRE_EXACT(S, GlobalSpace, ConstSpace)
	);

	InitializedVariableDeclaration(const std::string& name, const std::vector<typename T::SystemType>& initializer)
		: TypedVariableDeclaration<T, S>({name}), m_initializer(initializer) {}

	std::string ToString(unsigned int indentation, bool terminate) const override
	{
		auto code = TypedVariableDeclaration<T, S>::ToString(indentation, false);
		code += " = {";
		bool first = true;
		for (auto value : m_initializer)
		{
			if (!first)
			{
				code += ",";
			}
			first = false;
			code += " " + std::to_string(value);
		}
		code += " }";
		if (terminate)
		{
			code += ";";
		}
		return code;
	}

	json ToJSON() const override
	{
		json j = TypedVariableDeclaration<T, S>::ToJSON();
		j["initializer"] = m_initializer;
		return j;
	}

protected:
	std::vector<typename T::SystemType> m_initializer;
};

template<class T>
using RegisterDeclaration = TypedVariableDeclaration<T, RegisterSpace>;

template<class T>
using SpecialRegisterDeclaration = TypedVariableDeclaration<T, SpecialRegisterSpace>;

template<class T>
using LocalDeclaration = TypedVariableDeclaration<T, LocalSpace>;

template<class T>
using GlobalDeclaration = TypedVariableDeclaration<T, GlobalSpace>;

template<class T>
using InitializedGlobalDeclaration = InitializedVariableDeclaration<T, GlobalSpace>;

template<class T>
using SharedDeclaration = TypedVariableDeclaration<T, SharedSpace>;

template<class T>
using ConstDeclaration = TypedVariableDeclaration<T, ConstSpace>;

template<class T>
using InitializedConstDeclaration = InitializedVariableDeclaration<T, ConstSpace>;

template<class T>
using ParameterDeclaration = TypedVariableDeclaration<T, ParameterSpace>;

template<Bits B, class T, class S = AddressableSpace>
class PointerDeclaration : public ParameterDeclaration<PointerType<B, T, S>>
{
public:
	using ParameterDeclaration<PointerType<B, T, S>>::ParameterDeclaration;

	std::string PreDirectives() const override { return ""; } 

	std::string PostDirectives() const override
	{
		std::string code;
		if constexpr(!std::is_same<S, AddressableSpace>::value)
		{
			code += ".ptr" + S::Name() + ParameterDeclaration<PointerType<B, T, S>>::PreDirectives();
		}
		return code;
	}
};

template<class T, class S = AddressableSpace>
using Pointer32Declaration = PointerDeclaration<Bits::Bits32, T, S>;
template<class T, class S = AddressableSpace>
using Pointer64Declaration = PointerDeclaration<Bits::Bits64, T, S>;
}
