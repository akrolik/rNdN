#pragma once

#include "HorseIR/Traversal/ForwardTraversal.h"

#include "HorseIR/Tree/Expressions/CallExpression.h"
#include "HorseIR/Tree/Expressions/Identifier.h"
#include "HorseIR/Tree/Expressions/Literal.h"
#include "HorseIR/Tree/Statements/AssignStatement.h"

#include "PTX/Functions/Function.h"
#include "PTX/Instructions/Arithmetic/AddInstruction.h"
#include "PTX/Instructions/Data/MoveInstruction.h"
#include "PTX/Operands/Value.h"
#include "PTX/Operands/Variables/Register.h"

#include "Codegen/ExpressionGeneratorInterface.h"
#include "Codegen/OperandGenerator.h"
#include "Codegen/ResourceAllocator.h"

template<PTX::Bits B, class T>
class ExpressionGeneratorBase : public HorseIR::ForwardTraversal, public ExpressionGeneratorInterface<T>
{
public:
	ExpressionGeneratorBase(ResourceAllocator<B> *resources, const PTX::Register<T> *target, PTX::Function *function) : m_resources(resources), m_target(target), m_currentFunction(function) {}

	void Visit(HorseIR::CallExpression *call) override
	{
		std::string name = call->GetName();
		if (name == "@fill")
		{
			OperandGenerator<B, T> opGen(m_resources, m_currentFunction);
			auto src = opGen.GenerateOperand(call->GetArgument(1));
			this->GenerateMove(src);
		}
		else if (name == "@plus")
		{
			OperandGenerator<B, T> opGen(m_resources, m_currentFunction);
			auto src1 = opGen.GenerateOperand(call->GetArgument(0));
			auto src2 = opGen.GenerateOperand(call->GetArgument(1));
			this->GenerateAdd(src1, src2);
		}
	}

	std::enable_if_t<PTX::AddInstruction<T, false>::Enabled, void>
	GenerateAdd(const PTX::Operand<T> *src1, const PTX::Operand<T> *src2) override
	{
		m_currentFunction->AddStatement(new PTX::AddInstruction<T>(m_target, src1, src2));
	}

	std::enable_if_t<PTX::MoveInstruction<T, false>::Enabled, void>
	GenerateMove(const PTX::Operand<T> *src) override
	{
		m_currentFunction->AddStatement(new PTX::MoveInstruction<T>(m_target, src));
	}

private:
	ResourceAllocator<B> *m_resources = nullptr;
	const PTX::Register<T> *m_target = nullptr;
	PTX::Function *m_currentFunction = nullptr;
};
