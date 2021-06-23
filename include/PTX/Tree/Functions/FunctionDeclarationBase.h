#pragma once

#include "PTX/Tree/StateSpace.h"
#include "PTX/Tree/Type.h"
#include "PTX/Tree/Declarations/VariableDeclaration.h"
#include "PTX/Tree/Functions/Function.h"

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

	FunctionDeclarationBase(const std::string& name, ReturnDeclarationType *ret = nullptr, Declaration::LinkDirective linkDirective = Declaration::LinkDirective::None) : Function(name, linkDirective), m_return(ret) {}

	// Properties

	const ReturnDeclarationType *GetReturnDeclaration() const override { return m_return; }
	ReturnDeclarationType *GetReturnDeclaration() override { return m_return; }
	void SetReturnDeclaration(ReturnDeclarationType *ret) { m_return = ret; }

	std::string GetDirectives() const override
	{
		return ".func";
	}

	// Formatting

	json ToJSON() const override
	{
		json j = Function::ToJSON();
		j["return"] = m_return->ToJSON();
		return j;
	}

protected:
	ReturnDeclarationType *m_return = nullptr;
};

template<>
class FunctionDeclarationBase<VoidType> : public Function
{
public:
	using Function::Function;

	// Poperties

	bool GetEntry() const { return m_entry; }
	void SetEntry(bool entry) { m_entry = entry; }

	const VariableDeclaration *GetReturnDeclaration() const { return nullptr; }
	VariableDeclaration *GetReturnDeclaration() { return nullptr; }

	std::string GetDirectives() const override
	{
		if (m_entry)
		{
			return ".entry";
		}
		return ".func";
	}

protected:
	bool m_entry = false;
};

}
