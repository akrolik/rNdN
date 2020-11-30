#pragma once

#include "PTX/Tree/Instructions/PredicatedInstruction.h"
#include "PTX/Tree/Instructions/Modifiers/UniformModifier.h"

#include "PTX/Tree/Functions/Function.h"
#include "PTX/Tree/Operands/Extended/ListOperand.h"
#include "PTX/Tree/Operands/Extended/StringOperand.h"

namespace PTX {

template<class R>
class CallInstructionBase : public PredicatedInstruction, public UniformModifier
{
public:
	CallInstructionBase(const Function *function, const R *returnVariable, bool uniform) : UniformModifier(uniform), m_function(function), m_returnVariable(returnVariable) {}

	const Function *GetFunction() const { return m_function; }
	void SetFunction(const Function *function) { m_function = function; }

	const R *GetReturnVariable() const { return m_returnVariable; }
	void SetReturnVariable(const R *returnVariable) { m_returnVariable = returnVariable; }

	static std::string Mnemonic() { return "call"; }

	std::string OpCode() const override
	{
		return Mnemonic() + UniformModifier::OpCodeModifier();
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

	const Function *GetFunction() const { return m_function; }
	void SetFunction(const Function *function) { m_function = function; }

	static std::string Mnemonic() { return "call"; }

	std::string OpCode() const override
	{
		return Mnemonic() + UniformModifier::OpCodeModifier();
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
