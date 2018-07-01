#pragma once

#include "PTX/Instructions/PredicatedInstruction.h"
#include "PTX/Instructions/Modifiers/UniformModifier.h"

#include "PTX/Functions/Function.h"
#include "PTX/Operands/Extended/ListOperand.h"
#include "PTX/Operands/Extended/StringOperand.h"

namespace PTX {

template<class R>
class CallInstructionBase : public PredicatedInstruction, public UniformModifier
{
public:
	CallInstructionBase(const Function *function, const R *returnVariable, bool uniform) : UniformModifier(uniform), m_function(function), m_returnVariable(returnVariable) {}

	std::string OpCode() const override
	{
		return "call" + UniformModifier::OpCodeModifier();
	}

	std::vector<const Operand *> Operands() const override
	{
		std::vector<const Operand *> operands;
		operands.push_back(new ListOperand({ m_returnVariable }));
		operands.push_back(new StringOperand(m_function->GetName()));

		ListOperand *argumentList = new ListOperand();
		for (const auto& argument : GetArguments())
		{
			argumentList->AddOperand(argument);
		}
		operands.push_back(argumentList);
		return operands;
	}

protected:
	virtual std::vector<const Operand *> GetArguments() const = 0;

	const Function *m_function = nullptr;
	const R *m_returnVariable = nullptr;
};

template<>
class CallInstructionBase<VoidType> : public PredicatedInstruction, public UniformModifier
{
public:
	CallInstructionBase(const Function *function, bool uniform) : UniformModifier(uniform), m_function(function) {}

	std::string OpCode() const override
	{
		return "call" + UniformModifier::OpCodeModifier();
	}

	std::vector<const Operand *> Operands() const override
	{
		std::vector<const Operand *> operands;
		operands.push_back(new StringOperand(m_function->GetName()));

		ListOperand *argumentList = new ListOperand();
		for (const auto& argument : GetArguments())
		{
			argumentList->AddOperand(argument);
		}
		operands.push_back(argumentList);
		return operands;
	}

protected:
	virtual std::vector<const Operand *> GetArguments() const = 0;

	const Function *m_function = nullptr;
};

}
