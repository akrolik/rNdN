#pragma once

#include "PTX/Instructions/PredicatedInstruction.h"
#include "PTX/Instructions/Modifiers/UniformModifier.h"

#include "PTX/Functions/DataFunction.h"
#include "PTX/Utils.h"

namespace PTX {

template<class R, typename... Args>
class CallInstruction : public InstructionStatement, public UniformModifier
{
public:
	CallInstruction(DataFunction<R(Args...)> *function, R *returnVariable, Args* ...args, bool uniform = false) : UniformModifier(uniform), m_function(function), m_returnVariable(returnVariable), m_parameters(std::make_tuple(args...)) {}

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
		CodeTuple(code, " ", m_parameters, int_<sizeof...(Args)>());
		code << ")";

		return code.str();
	}

private:
	DataFunction<R(Args...)> *m_function = nullptr;
	R *m_returnVariable = nullptr;
	std::tuple<Args* ...> m_parameters;
};

template<typename... Args>
class CallInstruction<VoidType, Args...> : public InstructionStatement, public UniformModifier
{
public:
	CallInstruction(DataFunction<VoidType(Args...)> *function, Args* ...args, bool uniform = false) : UniformModifier(uniform), m_function(function), m_parameters(std::make_tuple(args...)) {}

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

		code << m_function->GetName() + ", (";
		CodeTuple(code, " ", m_parameters, int_<sizeof...(Args)>());
		code << ")";

		return code.str();
	}

private:
	DataFunction<VoidType(Args...)> *m_function = nullptr;
	std::tuple<Args* ...> m_parameters;
};

}
