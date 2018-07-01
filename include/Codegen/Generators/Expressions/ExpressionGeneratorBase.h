#pragma once

#include "Codegen/Generators/Expressions/ExpressionGeneratorInterface.h"
#include "HorseIR/Traversal/ForwardTraversal.h"

#include "HorseIR/Tree/Expressions/CallExpression.h"

#include "PTX/Instructions/Arithmetic/AddInstruction.h"
#include "PTX/Instructions/Data/MoveInstruction.h"
#include "PTX/Operands/Variables/Register.h"

#include "Codegen/Builder.h"
#include "Codegen/Generators/Expressions/OperandGenerator.h"

template<PTX::Bits B, class T>
class ExpressionGeneratorBase : public HorseIR::ForwardTraversal, public ExpressionGeneratorInterface<T>
{
public:
	ExpressionGeneratorBase(const PTX::Register<T> *target, Builder *builder) : m_target(target), m_builder(builder) {}

	void Visit(HorseIR::CallExpression *call) override
	{
		std::string name = call->GetName();
		if (name == "@fill")
		{
			OperandGenerator<B, T> opGen(m_builder);
			auto src = opGen.GenerateOperand(call->GetArgument(1));
			this->GenerateMove(src);
		}
		else if (name == "@plus")
		{
			OperandGenerator<B, T> opGen(m_builder);
			auto src1 = opGen.GenerateOperand(call->GetArgument(0));
			auto src2 = opGen.GenerateOperand(call->GetArgument(1));
			this->GenerateAdd(src1, src2);
		}
	}

	std::enable_if_t<PTX::AddInstruction<T, false>::TypeSupported, void>
	GenerateAdd(const PTX::TypedOperand<T> *src1, const PTX::TypedOperand<T> *src2) override
	{
		m_builder->AddStatement(new PTX::AddInstruction<T>(m_target, src1, src2));
	}

	std::enable_if_t<PTX::MoveInstruction<T, false>::TypeSupported, void>
	GenerateMove(const PTX::TypedOperand<T> *src) override
	{
		m_builder->AddStatement(new PTX::MoveInstruction<T>(m_target, src));
	}

protected:
	const PTX::Register<T> *m_target = nullptr;
	Builder *m_builder = nullptr;
};
