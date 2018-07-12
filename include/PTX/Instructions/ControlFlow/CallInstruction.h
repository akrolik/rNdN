#pragma once

#include "PTX/Instructions/ControlFlow/CallInstructionBase.h"

#include "PTX/Functions/FunctionDefinition.h"
#include "PTX/Operands/Variables/Variable.h"
#include "PTX/Tuple.h"

namespace PTX {

template<class R>
class CallInstruction : public CallInstructionBase<R>
{
public:
	CallInstruction(const FunctionDefinition<R> *function, const R *returnVariable, bool uniform = false) : CallInstructionBase<R>(function, returnVariable, uniform) {}

	template<class T, class S>
	std::enable_if_t<std::is_same<S, RegisterSpace>::value || std::is_base_of<S, ParameterSpace>::value, void>
	AddArgument(const Variable<T, S> *argument) { m_arguments.push_back(argument); }

protected:
	std::vector<const Operand *> GetArguments() const override
	{
		return m_arguments;
	}

	std::vector<const Operand *> m_arguments;
};

template<class R, typename... Args>
class CallInstruction<R(Args...)> : public CallInstructionBase<R>
{
public:
	CallInstruction(const FunctionDefinition<R(Args...)> *function, const R *returnVariable, const Args* ...args, bool uniform = false) : CallInstructionBase<R>(function, returnVariable, uniform), m_arguments(std::make_tuple(args...)) {}

protected:
	std::vector<const Operand *> GetArguments() const override
	{
		std::vector<const Operand *> arguments;
		ExpandTuple(arguments, m_arguments, int_<sizeof...(Args)>());
		return arguments;
	}

	template <typename T, size_t P>
	static std::vector<const Operand *>& ExpandTuple(std::vector<const Operand *>& operands, const T& t, int_<P>)
	{
		auto arg = std::get<std::tuple_size<T>::value-P>(t);
		if (arg == nullptr)
		{
			std::cerr << "[ERROR] Parameter " << std::tuple_size<T>::value-P << " not set" << std::endl;
			std::exit(EXIT_FAILURE);
		}
		operands.push_back(arg);
		return ExpandTuple(operands, t, int_<P-1>());
	}

	template <typename T>
	static std::vector<const Operand *>& ExpandTuple(std::vector<const Operand *>& operands, const T& t, int_<0>)
	{
		return operands;
	}

	std::tuple<const Args* ...> m_arguments;
};

}
