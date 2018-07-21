#pragma once

#include <tuple>
#include <sstream>

#include "PTX/StateSpace.h"
#include "PTX/Tuple.h"
#include "PTX/Type.h"
#include "PTX/Declarations/Declaration.h"
#include "PTX/Declarations/VariableDeclaration.h"
#include "PTX/Functions/FunctionDeclarationBase.h"

namespace PTX {

template<class R>
class FunctionDeclaration : public FunctionDeclarationBase<R>
{
public:
	using FunctionDeclarationBase<R>::FunctionDeclarationBase;
	using Signature = R;

	template<class T, class S>
	std::enable_if_t<REQUIRE_EXACT(S, RegisterSpace) || REQUIRE_BASE(S, ParameterSpace), void>
	AddParameter(const TypedVariableDeclaration<T, S> *parameter) { m_parameters.push_back(parameter); }

	json ToJSON() const override
	{
		json j = FunctionDeclarationBase<R>::ToJSON();
		for (const auto& parameter : m_parameters)
		{
			j["parameters"].push_back(parameter->ToJSON());
		}
		return j;
	}

protected:
	std::string GetParametersString() const override
	{
		std::string code;
		bool first = true;
		for (const auto& parameter : m_parameters)
		{
			if (first)
			{
				code += "\n";
			}
			else
			{
				code += ",\n";
			}
			first = false;
			code += parameter->ToString(1, false);
		}
		if (!first)
		{
			code += "\n";
		}
		return code;
	}

	std::vector<const VariableDeclaration *> m_parameters;
};

template<class R, typename... Args>
class FunctionDeclaration<R(Args...)> : public FunctionDeclarationBase<R>
{
public:
	REQUIRE_SPACE_PARAM(FunctionDeclaration,
		is_all<REQUIRE_EXACT(typename Args::VariableSpace, RegisterSpace) || REQUIRE_BASE(typename Args::VariableSpace, ParameterSpace)...>::value
	);

	using Signature = R(Args...);

	FunctionDeclaration() {}
	FunctionDeclaration(const std::string& name, const typename FunctionDeclarationBase<R>::ReturnDeclarationType *ret, const TypedVariableDeclaration<typename Args::VariableType, typename Args::VariableSpace>* ...parameters, Declaration::LinkDirective linkDirective = Declaration::LinkDirective::None) : FunctionDeclarationBase<R>(name, ret, linkDirective), m_parameters(std::make_tuple(parameters...)) {}

	void SetParameters(const TypedVariableDeclaration<typename Args::VariableType, typename Args::VariableSpace>* ...parameters) { m_parameters = std::make_tuple(parameters...); }

	json ToJSON() const override
	{
		json j = FunctionDeclarationBase<R>::ToJSON();
		std::vector<const Declaration *> parameters;
		ExpandTuple(parameters, m_parameters, int_<sizeof...(Args)>());
		for (const auto& parameter : parameters)
		{
			j["parameters"].push_back(parameter->ToJSON());
		}
		return j;
	}

protected:
	std::string GetParametersString() const override
	{
		std::ostringstream code;
		if constexpr(sizeof...(Args) > 0)
		{
			code << std::endl << "\t";
			CodeTuple(code, "\t\n", m_parameters, int_<sizeof...(Args)>(), 0, false);
		}
		return code.str();
	}

	template <typename T, size_t P>
	static std::vector<const Declaration *>& ExpandTuple(std::vector<const Declaration *>& declarations, const T& t, int_<P>)
	{
		auto arg = std::get<std::tuple_size<T>::value-P>(t);
		if (arg == nullptr)
		{
			std::cerr << "[ERROR] Parameter " << std::tuple_size<T>::value-P << " not set" << std::endl;
			std::exit(EXIT_FAILURE);
		}
		declarations.push_back(arg);
		return ExpandTuple(declarations, t, int_<P-1>());
	}

	template <typename T>
	static std::vector<const Declaration *>& ExpandTuple(std::vector<const Declaration *>& declarations, const T& t, int_<0>)
	{
		return declarations;
	}

	std::tuple<const TypedVariableDeclaration<typename Args::VariableType, typename Args::VariableSpace>* ...> m_parameters;
};

}
