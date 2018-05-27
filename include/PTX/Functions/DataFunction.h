#pragma once

#include <tuple>
#include <sstream>

#include "PTX/Type.h"
#include "PTX/Utils.h"
#include "PTX/Functions/Function.h"

#include "PTX/StateSpaces/StateSpace.h"
#include "PTX/StateSpaces/RegisterSpace.h"
#include "PTX/StateSpaces/AddressableSpace.h"

namespace PTX {

template<class R, typename... Args>
class DataFunction : public Function
{
	static_assert(std::is_same<RegisterSpace<typename R::SpaceType>, R>::value || std::is_base_of<ParameterSpace<typename R::SpaceType>, R>::value, "PTX::DataFunction return space must be a PTX::RegisterSpace or PTX::ParameterSpace");
	static_assert(is_all<std::is_same<RegisterSpace<typename Args::SpaceType>, Args>::value || std::is_base_of<ParameterSpace<typename Args::SpaceType>, Args>::value...>::value, "PTX::DataFunction parameter spaces must be PTX::RegisterSpaces or PTX::ParameterSpaces");
	
public:
	void SetReturnSpace(R *returnSpace) { m_returnSpace = returnSpace; }
	void SetParameters(Args* ...parameterSpaces) { m_parameterSpaces = std::make_tuple(parameterSpaces...); }

	std::string ToString() const
	{
		std::ostringstream code;

		code << GetDirectives() << " ";
		if (m_returnSpace != nullptr)
		{
			code << "(" << m_returnSpace->ToString() << ") ";
		}
		code << m_name << "(";
		if constexpr(sizeof...(Args) > 0)
		{
			code << std::endl << "\t";
			CodeTuple(code, "\t\n", m_parameterSpaces, int_<sizeof...(Args)>());
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
		if (m_visible)
		{
			return ".visible .func";
		}
		return ".func";
	}

private:
	R *m_returnSpace = nullptr;
	std::tuple<Args* ...> m_parameterSpaces;
};

template<typename... Args>
class DataFunction<VoidType, Args...> : public Function
{
	static_assert(is_all<std::is_same<RegisterSpace<typename Args::SpaceType>, Args>::value || std::is_base_of<ParameterSpace<typename Args::SpaceType>, Args>::value...>::value, "PTX::DataFunction parameter spaces must be PTX::RegisterSpaces or PTX::ParameterSpaces");
	
public:
	void SetParameters(Args* ...parameterSpaces) { m_parameterSpaces = std::make_tuple(parameterSpaces...); }

	std::string ToString() const
	{
		std::ostringstream code;

		code << GetDirectives() << " " << m_name << "(";
		if constexpr(sizeof...(Args) > 0)
		{
			code << std::endl << "\t";
			CodeTuple(code, "\t\n", m_parameterSpaces, int_<sizeof...(Args)>());
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
		if (m_visible)
		{
			return ".visible .func";
		}
		return ".func";
	}

private:
	std::tuple<Args* ...> m_parameterSpaces;
};
}
