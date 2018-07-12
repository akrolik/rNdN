#pragma once

#include "PTX/StateSpace.h"
#include "PTX/Type.h"
#include "PTX/Declarations/VariableDeclaration.h"
#include "PTX/Functions/Function.h"

namespace PTX {

template<class R>
class FunctionDefinitionBase : public Function
{
public:
	REQUIRE_SPACE_PARAM(FunctionDefinitionBase,
		REQUIRE_EXACT(typename R::VariableSpace, RegisterSpace) ||
		REQUIRE_BASE(typename R::VariableSpace, ParameterSpace)
	);

	void SetReturn(const VariableDeclaration<typename R::VariableType, typename R::VariableSpace> *ret) { m_return = ret; }

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
			return m_return->ToString();
		}
		return "<unset>";
	}

	const VariableDeclaration<typename R::VariableType, typename R::VariableSpace> *m_return = nullptr;
};

template<>
class FunctionDefinitionBase<VoidType> : public Function
{
public:
	bool GetEntry() const { return m_entry; }
	void SetEntry(bool entry) { m_entry = entry; }

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