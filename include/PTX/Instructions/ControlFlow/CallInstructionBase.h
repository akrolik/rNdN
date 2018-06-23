#pragma once

#include "PTX/Statements/InstructionStatement.h"
#include "PTX/Instructions/Modifiers/UniformModifier.h"

#include "PTX/Functions/Function.h"

namespace PTX {

template<class R>
class CallInstructionBase : public InstructionStatement, public UniformModifier
{
public:
	CallInstructionBase(const Function *function, const R *returnVariable, bool uniform) : UniformModifier(uniform), m_function(function), m_returnVariable(returnVariable) {}

	std::string OpCode() const override
	{
		return "call" + UniformModifier::OpCodeModifier();
	}

	std::string Operands() const override
	{
		return "(" + m_returnVariable->ToString() + "), " + m_function->GetName() + ", (" + GetArgumentsString() + ")";
	}

protected:
	virtual std::string GetArgumentsString() const = 0;

	const Function *m_function = nullptr;
	const R *m_returnVariable = nullptr;
};

template<>
class CallInstructionBase<VoidType> : public InstructionStatement, public UniformModifier
{
public:
	CallInstructionBase(const Function *function, bool uniform) : UniformModifier(uniform), m_function(function) {}

	std::string OpCode() const override
	{
		return "call" + UniformModifier::OpCodeModifier();
	}

	std::string Operands() const override
	{
		return m_function->GetName() + ", (" + GetArgumentsString() + ")";
	}

protected:
	virtual std::string GetArgumentsString() const = 0;

	const Function *m_function = nullptr;
};

}
