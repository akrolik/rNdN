#pragma once

#include "PTX/Tree/Instructions/ControlFlow/CallInstructionBase.h"

#include "PTX/Tree/Functions/FunctionDeclaration.h"
#include "PTX/Tree/Operands/Variables/Variable.h"
#include "PTX/Tree/Tuple.h"

#include "Utils/Logger.h"

namespace PTX {

template<class R>
class CallInstruction : public CallInstructionBase<R>
{
public:
	CallInstruction(FunctionDeclaration<R> *function, R *returnVariable, bool uniform = false)
		: CallInstructionBase<R>(function, returnVariable, uniform) {}

	// Properties

	template<class T, class S>
	std::enable_if_t<std::is_same<S, RegisterSpace>::value || std::is_base_of<S, ParameterSpace>::value, void>
	AddArgument(Variable<T, S> *argument) { m_arguments.push_back(argument); }

protected:
	const std::vector<Operand *>& GetArguments() const override { return m_arguments; }
	std::vector<Operand *> m_arguments;
};

template<class R, typename... Args>
class CallInstruction<R(Args...)> : public CallInstructionBase<R>
{
public:
	CallInstruction(FunctionDeclaration<R(Args...)> *function, R *returnVariable, Args* ...args, bool uniform = false)
		: CallInstructionBase<R>(function, returnVariable, uniform)
	{
		auto tuple = std::make_tuple(args...);
		ExpandTuple(m_arguments, tuple, int_<sizeof...(Args)>());
	}

protected:
	const std::vector<Operand *>& GetArguments() const override { return m_arguments; }
	std::vector<Operand *> m_arguments;
};

}
