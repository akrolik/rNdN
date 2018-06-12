#pragma once

#include <tuple>
#include <sstream>

#include "PTX/StateSpace.h"
#include "PTX/Type.h"
#include "PTX/Utils.h"
#include "PTX/Declarations/Declaration.h"
#include "PTX/Declarations/VariableDeclaration.h"
#include "PTX/Functions/Function.h"

namespace PTX {

template<class R, typename Enable = void>
class EntryModifier
{
protected:
	std::string GetDirectives() const
	{
		return ".func";
	}
};

template<class R>
class EntryModifier<R, std::enable_if_t<std::is_same<R, VoidType>::value>>
{
public:
	bool GetEntry() const { return m_entry; }
	void SetEntry(bool entry) { m_entry = entry; }

protected:
	std::string GetDirectives() const
	{
		if (m_entry)
		{
			return ".entry";
		}
		return ".func";
	}

	bool m_entry = false;
};

template<class R, typename Enable = void>
class ReturnParameter
{
protected:
	std::string GetReturnString() const
	{
		return "";
	}
};

template<class R>
class ReturnParameter<R, std::enable_if_t<!std::is_same<R, VoidType>::value>>
{
	static_assert(std::is_same<typename R::VariableSpace, RegisterSpace>::value || std::is_base_of<typename R::VariableSpace, ParameterSpace>::value, "PTX::DataFunction return space must be a PTX::RegisterSpace or PTX::ParameterSpace");
public:
	void SetReturn(VariableDeclaration<typename R::VariableType, typename R::VariableSpace> *ret) { m_return = ret; }

protected:
	std::string GetReturnString() const
	{
		if (m_return != nullptr)
		{
			return m_return->ToString();
		}
		return "<unset>";
	}

	VariableDeclaration<typename R::VariableType, typename R::VariableSpace> *m_return = nullptr;
};

class DynamicParameters
{
public:
	template<class T, class S>
	std::enable_if_t<std::is_same<S, RegisterSpace>::value || std::is_base_of<S, ParameterSpace>::value, void>
	AddParameter(VariableDeclaration<T, S>* parameter) { m_parameters.push_back(parameter); }

protected:
	std::string GetParametersString() const
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

template<typename... Args>
class StaticParameters
{
	static_assert(is_all<std::is_same<typename Args::VariableSpace, RegisterSpace>::value || std::is_base_of<typename Args::VariableSpace, ParameterSpace>::value...>::value, "PTX::DataFunction parameter spaces must be PTX::RegisterSpaces or PTX::ParameterSpaces");
public:
	void SetParameters(VariableDeclaration<typename Args::VariableType, typename Args::VariableSpace>* ...parameters) { m_parameters = std::make_tuple(parameters...); }

protected:
	std::string GetParametersString() const
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

template<class R>
class DataFunction : public Function, public EntryModifier<R>, public ReturnParameter<R>, public DynamicParameters
{
protected:
	std::string GetDirectives() const override { return EntryModifier<R>::GetDirectives(); }
	std::string GetReturnString() const override { return ReturnParameter<R>::GetReturnString(); }
	std::string GetParametersString() const override { return DynamicParameters::GetParametersString(); }
};

template<class R, typename... Args>
class DataFunction<R(Args...)> : public Function, public EntryModifier<R>, public ReturnParameter<R>, public StaticParameters<Args...>
{
protected:
	std::string GetDirectives() const override { return EntryModifier<R>::GetDirectives(); }
	std::string GetReturnString() const override { return ReturnParameter<R>::GetReturnString(); }
	std::string GetParametersString() const override { return StaticParameters<Args...>::GetParametersString(); }
};

}
