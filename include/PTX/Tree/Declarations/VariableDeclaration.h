#pragma once

#include <string>
#include <vector>

#include "PTX/Tree/Declarations/Declaration.h"
#include "PTX/Tree/Declarations/NameSet.h"

#include "PTX/Tree/Type.h"
#include "PTX/Tree/StateSpace.h"

#include "PTX/Traversal/Dispatch.h"

#include "Utils/Logger.h"

namespace PTX {

DispatchInterface_Space(TypedVariableDeclaration)

class VariableDeclaration : DispatchInherit(TypedVariableDeclaration), public Declaration
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

	// Properties

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
			if (set->GetName() == prefix)
			{
				set->SetCount(count);
				return;
			}
		}

		AddNames(prefix, count);
	}

	std::vector<const NameSet *> GetNames() const
	{
		return { std::begin(m_names), std::end(m_names) };
	}
	std::vector<NameSet *>& GetNames() { return m_names; }

	// Formatting

	virtual std::string PreDirectives() const { return ""; }
	virtual std::string PostDirectives() const { return ""; }

	virtual std::string ToString() const = 0;

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

	// Visitors

	void Accept(Visitor& visitor) override { visitor.Visit(this); }
	void Accept(ConstVisitor& visitor) const override { visitor.Visit(this); }

	void Accept(HierarchicalVisitor& visitor) override
	{
		visitor.VisitIn(this);
		visitor.VisitOut(this);
	}

	void Accept(ConstHierarchicalVisitor& visitor) const override
	{
		visitor.VisitIn(this);
		visitor.VisitOut(this);
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

	// Properties

	void SetAlignment(unsigned int alignment) { m_alignment = alignment; }
	unsigned int GetAlignment() const { return m_alignment; }

	// Formatting

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

template<class T, class S, bool Assert = true>
class TypedVariableDeclaration : public TypedVariableDeclarationBase<T, S>
{
public:
	REQUIRE_TYPE_PARAM(VariableDeclaration,
		REQUIRE_BASE(T, DataType)
	);

	REQUIRE_SPACE_PARAM(VariableDeclaration,
		REQUIRE_BASE(S, StateSpace) && !REQUIRE_EXACT(S, AddressableSpace)
	);

	using TypedVariableDeclarationBase<T, S>::TypedVariableDeclarationBase;

	// Properties

	typename S::template VariableType<T> *GetVariable(const std::string& name, unsigned int index = 0) const
	{
		for (const auto &set : this->m_names)
		{
			if (set->GetName() == name)
			{
				return new typename S::template VariableType<T>(set, index);
			}
		}
		Utils::Logger::LogError("PTX::Variable(" + name + ") not found in PTX::VariableDeclaration");
	}

	// Formatting

	std::string ToString() const override
	{
		std::string code;
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
		return code + " " + this->PostDirectives() + VariableNames();
	}

	json ToJSON() const override
	{
		json j = VariableDeclaration::ToJSON();
		j["type"] = T::Name();
		j["space"] = S::Name();
		return j;
	}

protected:
	DispatchMember_Type(T);
	DispatchMember_Space(S);

	std::string VariableNames() const
	{
		std::string code;
		auto first = true;
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

DispatchImplementation_Space(TypedVariableDeclaration)

//TODO: Initialized dispatch

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

	// Properties

	const std::vector<typename T::SystemType>& GetInitializer() const { return m_initializer; }
	void SetInitializer(const std::vector<typename T::SystemType>& initializer) { m_initializer = initializer; }

	// Formatting

	std::string ToString() const override
	{
		auto code = TypedVariableDeclaration<T, S>::ToString() + "{";
		auto first = true;
		for (auto value : m_initializer)
		{
			if (!first)
			{
				code += ",";
			}
			first = false;
			code += " " + std::to_string(value);
		}
		return code + " }";
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

	// Formatting

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
