#pragma once

#include <tuple>
#include <sstream>

#include "PTX/StateSpace.h"
#include "PTX/Type.h"
#include "PTX/Utils.h"
#include "PTX/Declarations/Declaration.h"
#include "PTX/Declarations/VariableDeclaration.h"
#include "PTX/Functions/ReturnedFunction.h"

namespace PTX {

template<class R>
class DataFunction : public ReturnedFunction<R>
{
public:
	template<class T, class S>
	std::enable_if_t<std::is_same<S, RegisterSpace>::value || std::is_base_of<S, ParameterSpace>::value, void>
	AddParameter(VariableDeclaration<T, S>* parameter) { m_parameters.push_back(parameter); }

protected:
	std::string GetParametersString() const override
	{
		std::ostringstream code;
		bool first = true;
		for (const auto& param : m_parameters)
		{
			if (!first)
			{
				code << ",";
			}
			first = false;
			code << std::endl << "\t" << param->ToString();
		}
		return code.str();
	}

	std::vector<Declaration *> m_parameters;
};

template<class R, typename... Args>
class DataFunction<R(Args...)> : public ReturnedFunction<R>
{
	static_assert(is_all<std::is_same<typename Args::VariableSpace, RegisterSpace>::value || std::is_base_of<typename Args::VariableSpace, ParameterSpace>::value...>::value, "PTX::DataFunction parameter spaces must be PTX::RegisterSpaces or PTX::ParameterSpaces");
public:
	void SetParameters(VariableDeclaration<typename Args::VariableType, typename Args::VariableSpace>* ...parameters) { m_parameters = std::make_tuple(parameters...); }

protected:
	std::string GetParametersString() const override
	{
		std::ostringstream code;
		if constexpr(sizeof...(Args) > 0)
		{
			code << std::endl << "\t";
			CodeTuple(code, "\t\n", m_parameters, int_<sizeof...(Args)>());
		}
		return code.str();
	}

	std::tuple<VariableDeclaration<typename Args::VariableType, typename Args::VariableSpace>* ...> m_parameters;
};

}
