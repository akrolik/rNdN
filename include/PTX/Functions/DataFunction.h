#pragma once

#include <tuple>
#include <sstream>

#include "PTX/StateSpace.h"
#include "PTX/Type.h"
#include "PTX/Utils.h"
#include "PTX/Functions/Function.h"

namespace PTX {

template<class R, typename... Args>
class DataFunction : public Function
{
	static_assert(std::is_same<typename R::VariableSpace, RegisterSpace>::value || std::is_base_of<typename R::VariableSpace, ParameterSpace>::value, "PTX::DataFunction return space must be a PTX::RegisterSpace or PTX::ParameterSpace");
	static_assert(is_all<std::is_same<typename Args::VariableSpace, RegisterSpace>::value || std::is_base_of<typename Args::VariableSpace, ParameterSpace>::value...>::value, "PTX::DataFunction parameter spaces must be PTX::RegisterSpaces or PTX::ParameterSpaces");
	
public:
	void SetReturn(VariableDeclaration<typename R::VariableType, typename R::VariableSpace> *ret) { m_return = ret; }
	void SetParameters(VariableDeclaration<typename Args::VariableType, typename Args::VariableSpace>* ...parameters) { m_parameters = std::make_tuple(parameters...); }

	std::string ToString() const
	{
		std::ostringstream code;

		if (m_linkDirective != LinkDirective::None)
		{
			code << LinkDirectiveString(m_linkDirective) << " ";
		}
		code << GetDirectives() << " ";
		if (m_return != nullptr)
		{
			code << "(" << m_return->ToString() << ") ";
		}
		code << m_name << "(";
		if constexpr(sizeof...(Args) > 0)
		{
			code << std::endl << "\t";
			CodeTuple(code, "\t\n", m_parameters, int_<sizeof...(Args)>());
		}
		code << ")" << std::endl;
		code << "{" << std::endl;
		if (m_body != nullptr)
		{
			code << m_body->ToString();
		}
		code << "}" << std::endl;
		
		return code.str();
	}

	virtual std::string GetDirectives() const
	{
		return ".func";
	}

private:
	VariableDeclaration<typename R::VariableType, typename R::VariableSpace> *m_return = nullptr;
	std::tuple<VariableDeclaration<typename Args::VariableType, typename Args::VariableSpace>* ...> m_parameters;
};

template<typename... Args>
class DataFunction<VoidType, Args...> : public Function
{
	static_assert(is_all<std::is_same<typename Args::VariableSpace, RegisterSpace>::value || std::is_base_of<typename Args::VariableSpace, ParameterSpace>::value...>::value, "PTX::DataFunction parameter spaces must be PTX::RegisterSpaces or PTX::ParameterSpaces");
	
public:
	void SetParameters(VariableDeclaration<typename Args::VariableType, typename Args::VariableSpace>* ...parameters) { m_parameters = std::make_tuple(parameters...); }

	std::string ToString() const
	{
		std::ostringstream code;

		if (m_linkDirective != LinkDirective::None)
		{
			code << LinkDirectiveString(m_linkDirective) << " ";
		}
		code << GetDirectives() << " " << m_name << "(";
		if constexpr(sizeof...(Args) > 0)
		{
			code << std::endl << "\t";
			CodeTuple(code, "\t\n", m_parameters, int_<sizeof...(Args)>());
		}
		code << ")" << std::endl;
		code << "{" << std::endl;
		if (m_body != nullptr)
		{
			code << m_body->ToString();
		}
		code << "}" << std::endl;
		
		return code.str();
	}

	virtual std::string GetDirectives() const
	{
		return ".func";
	}

private:
	std::tuple<VariableDeclaration<typename Args::VariableType, typename Args::VariableSpace>* ...> m_parameters;
};
}
