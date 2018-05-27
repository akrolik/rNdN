#pragma once

#include "PTX/Instructions/PredicatedInstruction.h"
#include "PTX/Instructions/Modifiers/UniformModifier.h"

#include "PTX/Functions/DataFunction.h"

namespace PTX {

template<class R, typename... Args>
class CallInstruction : public InstructionStatement, public UniformModifier
{
public:
	CallInstruction(DataFunction<R, Args...> *function, Variable<typename R::SpaceType, R> *returnVariable, Variable<typename Args::SpaceType, Args>* ...args, bool uniform = false) : UniformModifier(uniform), m_function(function), m_returnVariable(returnVariable), m_parameters(std::make_tuple(args...)) {}

	std::string OpCode() const
	{
		if (m_uniform)
		{
			return "call.uni";
		}
		return "call";
	}

	std::string Operands() const
	{
		std::ostringstream code;

		code << "(" + m_returnVariable->ToString() + "), " + m_function->GetName() + ", (";
		CodeArgs(code, m_parameters, int_<sizeof...(Args)>());
		code << ")";

		return code.str();
	}

private:
	DataFunction<R, Args...> *m_function = nullptr;
	Variable<typename R::SpaceType, R> *m_returnVariable = nullptr;
	std::tuple<Variable<typename Args::SpaceType, Args>* ...> m_parameters;

	template<std::size_t> struct int_{};

	template <typename T, size_t P>
	std::ostringstream& CodeArgs(std::ostringstream& code, const T& t, int_<P>) const
	{
		auto arg = std::get<std::tuple_size<T>::value-P>(t);
		if (arg == nullptr)
		{
			std::cerr << "[Error] Parameter " << std::tuple_size<T>::value-P << " not set in call" << std::endl;
			std::exit(EXIT_FAILURE);
		}
		code << arg->ToString() << ", ";
		return CodeArgs(code, t, int_<P-1>());
	}

	template <typename T>
	std::ostringstream& CodeArgs(std::ostringstream& code, const T& t, int_<1>) const
	{
		auto arg = std::get<std::tuple_size<T>::value-1>(t);
		if (arg == nullptr)
		{
			std::cerr << "[Error] Parameter " << std::tuple_size<T>::value-1 << " not set in call" << std::endl;
			std::exit(EXIT_FAILURE);
		}
		code << arg->ToString();
		return code;
	}

	template <typename T>
	std::ostringstream& CodeArgs(std::ostringstream& code, const T& t, int_<0>) const
	{
		return code;
	}
};

}
