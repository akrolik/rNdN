#pragma once

#include <tuple>
#include <sstream>

#include "PTX/Type.h"

#include "PTX/Functions/Function.h"

#include "PTX/StateSpaces/StateSpace.h"
#include "PTX/StateSpaces/PointerSpace.h"

namespace PTX {

// template <class T, template <typename> class Template>
// struct is_space_specialization : std::false_type {};

// template <template <typename> class Template, class Args>
// struct is_space_specialization<Template<Args>, Template> : std::true_type {};

// template <bool... B> struct is_all;

// template <bool... T>
// struct is_all<true, T...> : is_all<T...> {};

// template <bool... T>
// struct is_all<false, T...> : std::false_type {};

// template <> struct is_all<> : std::true_type {};

template<class R, typename... Args>
class DataFunction : public Function
{
	// static_assert(std::is_base_of<DataType, R>::value || std::is_same<VoidType, R>::value, "PTX::DataFunction return type must be a PTX::DataType or PTX::VoidType");
	// static_assert(is_all<is_space_specialization<Args, StateSpace>::value...>::value, "Args must be PTX::StateSpaces");
	
public:
	void SetReturnSpace(R *returnSpace) { m_returnSpace = returnSpace; }
	// template<class Q=R>
	// void SetReturnSpace(std::enable_if_t<!std::is_same<Q, VoidType>::value, StateSpace<Q>> *returnSpace) { m_returnSpace = returnSpace; }
	// void SetReturnSpace(std::enable_if_t<!std::is_same<Q, VoidType>::value, StateSpace<Q>> *returnSpace) { m_returnSpace = returnSpace; }
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
			code << std::endl;
		}
		CodeArgs(code, m_parameterSpaces, int_<sizeof...(Args)>());
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
		code << "\t" << arg->ToString() << "," << std::endl;
		return CodeArgs(code, t, int_<P-1>());
	}

	template <typename T>
	std::ostringstream& CodeArgs(std::ostringstream& code, const T& t, int_<1>) const
	{
		auto arg = std::get<std::tuple_size<T>::value-1>(t);
		if (arg == nullptr)
		{
			std::cerr << "[Error] Parameter " << std::tuple_size<T>::value-1 << " not set in function " << m_name << std::endl;
			std::exit(EXIT_FAILURE);
		}
		code << "\t" << arg->ToString() << std::endl;
		return code;
	}

	template <typename T>
	std::ostringstream& CodeArgs(std::ostringstream& code, const T& t, int_<0>) const
	{
		return code;
	}
};

}
