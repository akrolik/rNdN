#pragma once

#include <tuple>
#include <sstream>

#include "PTX/Functions/Function.h"

#include "PTX/StateSpaces/StateSpace.h"

namespace PTX {

// template <bool... B> struct All;

// template <bool... T>
// struct All<true, T...> : All<T...> {};

// template <bool... T>
// struct All<false, T...> : std::false_type {};

// template <> struct All<> : std::true_type {};

template<class R, typename... Args>
class DataFunction : public Function
{
	static_assert(std::is_base_of<ValueType, R>::value || std::is_same<VoidType, R>::value, "T must be a PTX::ValueType or PTX::VoidType");
	// static_assert(All<std::is_base_of<StateSpace, Args>::value...>::value, "Args must be PTX::ValueTypes");
	
public:
	template<class Q=R>
	void SetReturnSpace(std::enable_if_t<std::is_same<Q, VoidType>::value, StateSpace<Q>> *returnSpace) { m_returnSpace = returnSpace; }
	void SetParameters(Args* ...parameterSpaces) { m_parameterSpaces = std::make_tuple(parameterSpaces...); }

	std::string ToString() const
	{
		std::ostringstream code;

		code << GetDirectives() << " ";
		
		if (m_returnSpace != nullptr)
		{
			code << "(" << m_returnSpace->ToString() << ") ";
		}

		code << m_name << "(" << std::endl;
		CodeArgs(code, m_parameterSpaces, int_<sizeof...(Args)>());
		code << ")" << std::endl;
		code << "{" << std::endl;
		code << m_body->ToString();
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
	StateSpace<R> *m_returnSpace = nullptr;
	std::tuple<Args* ...> m_parameterSpaces;

	template<std::size_t> struct int_{};

	template <typename T, size_t P>
	std::ostringstream& CodeArgs(std::ostringstream& code, const T& t, int_<P>) const
	{
		auto arg = std::get<std::tuple_size<T>::value-P>(t);
		if (arg == nullptr)
		{
			std::cerr << "[Error] Parameter " << std::tuple_size<T>::value-P << " not set in function " << m_name << std::endl;
			std::exit(EXIT_FAILURE);
		}
		code << arg->ToString();
		return CodeArgs(code, t, int_<P-1>());
	}

	template <typename T>
	std::ostringstream& CodeArgs(std::ostringstream& code, const T& t, int_<0>) const
	{
		return code;
	}
};

}
