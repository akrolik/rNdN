#pragma once

#include "PTX/Tree/Functions/Function.h"

#include "PTX/Tree/Type.h"
#include "PTX/Tree/Declarations/VariableDeclaration.h"

namespace PTX {

template<class RT, class RS>
class FunctionDeclarationBase : public Function
{
public:
	using ReturnDeclarationType = TypedVariableDeclaration<RT, RS>;

	FunctionDeclarationBase(const std::string& name, ReturnDeclarationType *ret = nullptr, Declaration::LinkDirective linkDirective = Declaration::LinkDirective::None) : Function(name, linkDirective), m_return(ret) {}

	// Properties

	const ReturnDeclarationType *GetReturnDeclaration() const { return m_return; }
	ReturnDeclarationType *GetReturnDeclaration() { return m_return; }
	void SetReturnDeclaration(ReturnDeclarationType *ret) { m_return = ret; }

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

template<class RS>
class FunctionDeclarationBase<VoidType, RS> : public Function
{
public:
	using Function::Function;

	// Poperties

	bool IsEntry() const { return m_entry; }
	void SetEntry(bool entry) { m_entry = entry; }

	bool IsNoreturn() const { return m_noreturn; }
	void SetNoreturn(bool noret) { m_noreturn = noret; }

	// Formatting

	json ToJSON() const override
	{
		json j = Function::ToJSON();
		if (m_entry)
		{
			j["entry"] = true;
		}
		if (m_noreturn)
		{
			j["no_entry"] = true;
		}
		return j;
	}

protected:
	bool m_entry = false;
	bool m_noreturn = false;
};

}
