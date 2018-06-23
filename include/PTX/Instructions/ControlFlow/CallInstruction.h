#pragma once

#include "PTX/Instructions/ControlFlow/CallInstructionBase.h"

#include "PTX/Functions/DataFunction.h"
#include "PTX/Operands/Variables/Variable.h"
#include "PTX/Tuple.h"

namespace PTX {


template<class R>
class CallInstruction : public CallInstructionBase<R>
{
public:
	CallInstruction(const DataFunction<R> *function, const R *returnVariable, bool uniform = false) : CallInstructionBase<R>(function, returnVariable, uniform) {}

	template<class T, class S>
	std::enable_if_t<std::is_same<S, RegisterSpace>::value || std::is_base_of<S, ParameterSpace>::value, void>
	AddArgument(const Variable<T, S> *argument) { m_arguments.push_back(argument); }

protected:
	std::string GetArgumentsString() const override
	{
		std::string code;
		bool first = true;
		for (const auto& argument : m_arguments)
		{
			if (!first)
			{
				code += ",";
			}
			first = false;
			code += argument->ToString();
		}
		return code;
	}

	std::vector<const UntypedVariable*> m_arguments;
};

template<class R, typename... Args>
class CallInstruction<R(Args...)> : public CallInstructionBase<R>
{
public:
	CallInstruction(const DataFunction<R(Args...)> *function, const R *returnVariable, const Args* ...args, bool uniform = false) : CallInstructionBase<R>(function, returnVariable, uniform), m_arguments(std::make_tuple(args...)) {}

protected:
	std::string GetArgumentsString() const override
	{
		std::ostringstream code;
		CodeTuple(code, " ", m_arguments, int_<sizeof...(Args)>());
		return code.str();
	}

	std::tuple<const Args* ...> m_arguments;
};

}
