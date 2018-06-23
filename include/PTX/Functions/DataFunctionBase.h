#pragma once

#include "PTX/StateSpace.h"
#include "PTX/Type.h"
#include "PTX/Declarations/VariableDeclaration.h"
#include "PTX/Functions/Function.h"

namespace PTX {

template<class R>
class DataFunctionBase : public Function
{
	static_assert(std::is_same<typename R::VariableSpace, RegisterSpace>::value || std::is_base_of<typename R::VariableSpace, ParameterSpace>::value, "PTX::DataFunction return space must be a PTX::RegisterSpace or PTX::ParameterSpace");
public:
	void SetReturn(const VariableDeclaration<typename R::VariableType, typename R::VariableSpace> *ret) { m_return = ret; }

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
class DataFunctionBase<VoidType> : public Function
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
