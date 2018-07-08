#pragma once

#include "HorseIR/Traversal/ForwardTraversal.h"

#include "HorseIR/Tree/Expressions/CallExpression.h"

#include "PTX/Operands/Variables/Register.h"

#include "Codegen/Builder.h"
#include "Codegen/Generators/Expressions/Builtins/BuiltinGenerator.h"
#include "Codegen/Generators/Expressions/Builtins/BinaryGenerator.h"
#include "Codegen/Generators/Expressions/Builtins/ComparisonGenerator.h"
#include "Codegen/Generators/Expressions/Builtins/CompressionGenerator.h"
#include "Codegen/Generators/Expressions/Builtins/FillGenerator.h"
#include "Codegen/Generators/Expressions/Builtins/ReductionGenerator.h"
#include "Codegen/Generators/Expressions/Builtins/TrigonometryGenerator.h"

namespace Codegen {

template<PTX::Bits B, class T>
class ExpressionGenerator : public HorseIR::ForwardTraversal
{
public:
	ExpressionGenerator(const PTX::Register<T> *target, Builder *builder) : m_target(target), m_builder(builder) {}

	void Visit(HorseIR::CallExpression *call) override
	{
		BuiltinGenerator<B, T> *generator = GetBuiltinGenerator(call);
		if (generator == nullptr)
		{
			std::cerr << "[ERROR] Generator for builtin function " + call->GetName() + " not implemented" << std::endl;
			std::exit(EXIT_FAILURE);
		}
		generator->Generate(call);
		delete generator;
	}

	BuiltinGenerator<B, T> *GetBuiltinGenerator(const HorseIR::CallExpression *call)
	{
		//TODO: Complete adding generator instantiations
		std::string name = call->GetName();
		if (name == "@abs")
		{
		}
		else if (name == "@neg")
		{
		}
		else if (name == "@ceil")
		{
		}
		else if (name == "@floor")
		{
		}
		else if (name == "@round")
		{
		}
		else if (name == "@conj")
		{
		}
		else if (name == "@recip")
		{
		}
		else if (name == "@signum")
		{
		}
		else if (name == "@pi")
		{
		}
		else if (name == "@not")
		{
		}
		else if (name == "@log")
		{
		}
		else if (name == "@exp")
		{
		}
		else if (name == "@cos")
		{
			return new TrigonometryGenerator<B, T>(m_target, m_builder, TrigonometryOperation::Cosine);
		}
		else if (name == "@sin")
		{
			return new TrigonometryGenerator<B, T>(m_target, m_builder, TrigonometryOperation::Sine);
		}
		else if (name == "@tan")
		{
			return new TrigonometryGenerator<B, T>(m_target, m_builder, TrigonometryOperation::Tangent);
		}
		else if (name == "@acos")
		{
			return new TrigonometryGenerator<B, T>(m_target, m_builder, TrigonometryOperation::InverseCosine);
		}
		else if (name == "@asin")
		{
			return new TrigonometryGenerator<B, T>(m_target, m_builder, TrigonometryOperation::InverseSine);
		}
		else if (name == "@atan")
		{
			return new TrigonometryGenerator<B, T>(m_target, m_builder, TrigonometryOperation::Tangent);
		}
		else if (name == "@cosh")
		{
			return new TrigonometryGenerator<B, T>(m_target, m_builder, TrigonometryOperation::HyperbolicCosine);
		}
		else if (name == "@sinh")
		{
			return new TrigonometryGenerator<B, T>(m_target, m_builder, TrigonometryOperation::HyperbolicSine);
		}
		else if (name == "@tanh")
		{
			return new TrigonometryGenerator<B, T>(m_target, m_builder, TrigonometryOperation::HyperbolicTangent);
		}
		else if (name == "@acosh")
		{
			return new TrigonometryGenerator<B, T>(m_target, m_builder, TrigonometryOperation::HyperbolicInverseCosine);
		}
		else if (name == "@asinh")
		{
			return new TrigonometryGenerator<B, T>(m_target, m_builder, TrigonometryOperation::HyperbolicInverseSine);
		}
		else if (name == "@atanh")
		{
			return new TrigonometryGenerator<B, T>(m_target, m_builder, TrigonometryOperation::HyperbolicInverseTangent);
		}
		else if (name == "@lt")
		{
			return new ComparisonGenerator<B, T>(m_target, m_builder, ComparisonOperator::Less);
		}
		else if (name == "@gt")
		{
			return new ComparisonGenerator<B, T>(m_target, m_builder, ComparisonOperator::Greater);
		}
		else if (name == "@leq")
		{
			return new ComparisonGenerator<B, T>(m_target, m_builder, ComparisonOperator::LessEqual);
		}
		else if (name == "@geq")
		{
			return new ComparisonGenerator<B, T>(m_target, m_builder, ComparisonOperator::GreaterEqual);
		}
		else if (name == "@eq")
		{
			return new ComparisonGenerator<B, T>(m_target, m_builder, ComparisonOperator::Equal);
		}
		else if (name == "@neq")
		{
			return new ComparisonGenerator<B, T>(m_target, m_builder, ComparisonOperator::NotEqual);
		}
		else if (name == "@plus")
		{
			return new BinaryGenerator<B, T>(m_target, m_builder, BinaryOperation::Plus);
		}
		else if (name == "@minus")
		{
			return new BinaryGenerator<B, T>(m_target, m_builder, BinaryOperation::Minus);
		}
		else if (name == "@mul")
		{
			return new BinaryGenerator<B, T>(m_target, m_builder, BinaryOperation::Multiply);
		}
		else if (name == "@div")
		{
			return new BinaryGenerator<B, T>(m_target, m_builder, BinaryOperation::Divide);
		}
		else if (name == "@power")
		{
		}
		else if (name == "@log2")
		{
		}
		else if (name == "@mod")
		{
		}
		else if (name == "@and")
		{
			return new BinaryGenerator<B, T>(m_target, m_builder, BinaryOperation::And);
		}
		else if (name == "@or")
		{
			return new BinaryGenerator<B, T>(m_target, m_builder, BinaryOperation::Or);
		}
		else if (name == "@nand")
		{
			return new BinaryGenerator<B, T>(m_target, m_builder, BinaryOperation::Nand);
		}
		else if (name == "@nor")
		{
			return new BinaryGenerator<B, T>(m_target, m_builder, BinaryOperation::Nor);
		}
		else if (name == "@xor")
		{
			return new BinaryGenerator<B, T>(m_target, m_builder, BinaryOperation::Xor);
		}
		else if (name == "@compress")
		{
			return new CompressionGenerator<B, T>(m_target, m_builder);
		}
		else if (name == "@count")
		{
			return new ReductionGenerator<B, T>(m_target, m_builder, ReductionOperation::Count);
		}
		else if (name == "@sum")
		{
			return new ReductionGenerator<B, T>(m_target, m_builder, ReductionOperation::Sum);
		}
		else if (name == "@avg")
		{
			return new ReductionGenerator<B, T>(m_target, m_builder, ReductionOperation::Average);
		}
		else if (name == "@min")
		{
			return new ReductionGenerator<B, T>(m_target, m_builder, ReductionOperation::Minimum);
		}
		else if (name == "@max")
		{
			return new ReductionGenerator<B, T>(m_target, m_builder, ReductionOperation::Maximum);
		}
		else if (name == "@fill")
		{
			return new FillGenerator<B, T>(m_target, m_builder);
		}
		return nullptr;
	}

protected:
	const PTX::Register<T> *m_target = nullptr;
	Builder *m_builder = nullptr;
};

}
