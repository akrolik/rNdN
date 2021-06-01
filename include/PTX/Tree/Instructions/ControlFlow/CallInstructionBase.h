#pragma once

#include "PTX/Tree/Instructions/PredicatedInstruction.h"
#include "PTX/Tree/Instructions/Modifiers/UniformModifier.h"

#include "PTX/Tree/Functions/Function.h"
#include "PTX/Tree/Operands/Extended/ListOperand.h"
#include "PTX/Tree/Operands/Extended/StringOperand.h"

namespace PTX {

DispatchInterface(CallInstructionBase)

template<class R, bool Assert = true>
class CallInstructionBase : DispatchInherit(CallInstructionBase), public PredicatedInstruction, public UniformModifier
{
public:
	CallInstructionBase(Function *function, R *returnVariable, bool uniform) : UniformModifier(uniform), m_function(function), m_returnVariable(returnVariable) {}

	// Analysis properties

	bool HasSideEffect() const override { return true; }

	// Properties

	const Function *GetFunction() const { return m_function; }
	Function *GetFunction() { return m_function; }
	void SetFunction(Function *function) { m_function = function; }

	const R *GetReturnVariable() const { return m_returnVariable; }
	R *GetReturnVariable() { return m_returnVariable; }
	void SetReturnVariable(R *returnVariable) { m_returnVariable = returnVariable; }

	// Formatting

	static std::string Mnemonic() { return "call"; }

	std::string GetOpCode() const override
	{
		return Mnemonic() + UniformModifier::GetOpCodeModifier();
	}

	std::vector<const Operand *> GetOperands() const override
	{
		std::vector<const Operand *> operands;
		operands.push_back(new ListOperand({ m_returnVariable }));
		operands.push_back(new StringOperand(m_function->GetName()));

		auto argumentList = new ListOperand();
		for (auto& argument : GetArguments())
		{
			argumentList->AddOperand(argument);
		}
		operands.push_back(argumentList);
		return operands;
	}

	std::vector<Operand *> GetOperands() override
	{
		std::vector<Operand *> operands;
		operands.push_back(new ListOperand({ m_returnVariable }));
		operands.push_back(new StringOperand(m_function->GetName()));

		auto argumentList = new ListOperand();
		for (auto& argument : GetArguments())
		{
			argumentList->AddOperand(argument);
		}
		operands.push_back(argumentList);
		return operands;
	}

	// Visitors

	void Accept(InstructionVisitor& visitor) override { visitor.Visit(this); }
	void Accept(ConstInstructionVisitor& visitor) const override { visitor.Visit(this); }

protected:
	DispatchMember_Type(typename R::VariableType);

	virtual const std::vector<Operand *>& GetArguments() const = 0;

	Function *m_function = nullptr;
	R *m_returnVariable = nullptr;
};

template<bool Assert>
class CallInstructionBase<VoidType, Assert> : DispatchInherit(CallInstructionBase), public PredicatedInstruction, public UniformModifier
{
public:
	CallInstructionBase(Function *function, bool uniform) : UniformModifier(uniform), m_function(function) {}

	// Properties

	const Function *GetFunction() const { return m_function; }
	Function *GetFunction() { return m_function; }
	void SetFunction(Function *function) { m_function = function; }

	// Formatting

	static std::string Mnemonic() { return "call"; }

	std::string GetOpCode() const override
	{
		return Mnemonic() + UniformModifier::GetOpCodeModifier();
	}

	std::vector<const Operand *> GetOperands() const override
	{
		std::vector<const Operand *> operands;
		operands.push_back(new StringOperand(m_function->GetName()));

		auto argumentList = new ListOperand();
		for (const auto& argument : GetArguments())
		{
			argumentList->AddOperand(argument);
		}
		operands.push_back(argumentList);
		return operands;
	}

	std::vector<Operand *> GetOperands() override
	{
		std::vector<Operand *> operands;
		operands.push_back(new StringOperand(m_function->GetName()));

		auto argumentList = new ListOperand();
		for (auto& argument : GetArguments())
		{
			argumentList->AddOperand(argument);
		}
		operands.push_back(argumentList);
		return operands;
	}

	// Visitors

	void Accept(InstructionVisitor& visitor) override { visitor.Visit(this); }
	void Accept(ConstInstructionVisitor& visitor) const override { visitor.Visit(this); }

protected:
	DispatchMember_Type(VoidType);

	virtual const std::vector<Operand *>& GetArguments() const = 0;

	Function *m_function = nullptr;
};

}
