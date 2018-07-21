#pragma once

#include "PTX/StateSpace.h"
#include "PTX/Type.h"
#include "PTX/Declarations/VariableDeclaration.h"
#include "PTX/Functions/Function.h"

namespace PTX {

template<class R>
class FunctionDeclarationBase : public Function
{
public:
	REQUIRE_SPACE_PARAM(FunctionDeclarationBase,
		REQUIRE_EXACT(typename R::VariableSpace, RegisterSpace) ||
		REQUIRE_BASE(typename R::VariableSpace, ParameterSpace)
	);

	using ReturnDeclarationType = TypedVariableDeclaration<typename R::VariableType, typename R::VariableSpace>;

	FunctionDeclarationBase() {}
	FunctionDeclarationBase(const std::string& name, const ReturnDeclarationType *ret = nullptr, Declaration::LinkDirective linkDirective = Declaration::LinkDirective::None) : Function(name, linkDirective), m_return(ret) {}

	void SetReturn(const ReturnDeclarationType *ret) { m_return = ret; }

	std::string ToString(unsigned int indentation = 0) const override
	{
		return Function::ToString(indentation) + ";";
	}

	json ToJSON() const override
	{
		json j = Function::ToJSON();
		j["return"] = m_return->ToJSON();
		return j;
	}

protected:
	std::string GetDirectives() const override
	{
		return ".func";
	}

	std::string GetReturnString() const override
	{
		if (m_return != nullptr)
		{
			return m_return->ToString(0, false);
		}
		return "<unset>";
	}

	const ReturnDeclarationType *m_return = nullptr;
};

template<>
class FunctionDeclarationBase<VoidType> : public Function
{
public:
	FunctionDeclarationBase() {}
	FunctionDeclarationBase(const std::string& name, Declaration::LinkDirective linkDirective) : Function(name) {}

	bool GetEntry() const { return m_entry; }
	void SetEntry(bool entry) { m_entry = entry; }

	std::string ToString(unsigned int indentation = 0) const override
	{
		return Function::ToString(indentation) + ";";
	}

protected:
	std::string GetDirectives() const override
	{
		if (m_entry)
		{
			return ".entry";
		}
		return ".func";
	}

	std::string GetReturnString() const override
	{
		return "";
	}

	bool m_entry = false;
};

}
