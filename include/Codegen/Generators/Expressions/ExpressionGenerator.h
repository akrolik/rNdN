#pragma once

#include "HorseIR/Traversal/ForwardTraversal.h"
#include "Codegen/Generators/Generator.h"

#include "HorseIR/Tree/Expressions/CallExpression.h"

#include "PTX/Operands/Variables/Register.h"

#include "Codegen/Generators/Expressions/Builtins/BuiltinGenerator.h"
#include "Codegen/Generators/Expressions/Builtins/BinaryGenerator.h"
#include "Codegen/Generators/Expressions/Builtins/ComparisonGenerator.h"
#include "Codegen/Generators/Expressions/Builtins/CompressionGenerator.h"
#include "Codegen/Generators/Expressions/Builtins/ExternalBinaryGenerator.h"
#include "Codegen/Generators/Expressions/Builtins/ExternalUnaryGenerator.h"
#include "Codegen/Generators/Expressions/Builtins/FillGenerator.h"
#include "Codegen/Generators/Expressions/Builtins/ReductionGenerator.h"
#include "Codegen/Generators/Expressions/Builtins/RoundingGenerator.h"
#include "Codegen/Generators/Expressions/Builtins/UnaryGenerator.h"

namespace Codegen {

template<PTX::Bits B, class T>
class ExpressionGenerator : public HorseIR::ForwardTraversal, public Generator
{
public:
	ExpressionGenerator(const std::string& target, Builder *builder) : Generator(builder), m_target(target) {}

	void Visit(HorseIR::CallExpression *call) override
	{
		BuiltinGenerator<B, T> *generator = GetBuiltinGenerator(call);
		if (generator == nullptr)
		{
			std::cerr << "[ERROR] Generator for builtin function " + call->GetName() + " not implemented" << std::endl;
			std::exit(EXIT_FAILURE);
		}
		generator->Generate(m_target, call);
		delete generator;
	}

	BuiltinGenerator<B, T> *GetBuiltinGenerator(const HorseIR::CallExpression *call)
	{
		BuiltinFunction function = GetBuiltinFunction(call->GetName());
		switch (function)
		{
			case BuiltinFunction::Absolute:
				return new UnaryGenerator<B, T>(this->m_builder, UnaryOperation::Absolute);
			case BuiltinFunction::Negate:
				return new UnaryGenerator<B, T>(this->m_builder, UnaryOperation::Negate);
			case BuiltinFunction::Ceiling:
				return new RoundingGenerator<B, T>(this->m_builder, RoundingOperation::Ceiling);
			case BuiltinFunction::Floor:
				return new RoundingGenerator<B, T>(this->m_builder, RoundingOperation::Floor);
			case BuiltinFunction::Round:
				return new RoundingGenerator<B, T>(this->m_builder, RoundingOperation::Nearest);
			case BuiltinFunction::Conjugate:
				//TODO: Add support for complex numbers
				std::cerr << "[ERROR] Complex number functions are not supported" << std::endl;
				std::exit(EXIT_FAILURE);
			case BuiltinFunction::Reciprocal:
				return new UnaryGenerator<B, T>(this->m_builder, UnaryOperation::Reciprocal);
			case BuiltinFunction::Sign:
				return new ComparisonGenerator<B, T>(this->m_builder, ComparisonOperator::Sign);
			case BuiltinFunction::Pi:
				return new UnaryGenerator<B, T>(this->m_builder, UnaryOperation::Pi);
			case BuiltinFunction::Not:
				return new UnaryGenerator<B, T>(this->m_builder, UnaryOperation::Not);
			case BuiltinFunction::Logarithm:
				return new ExternalUnaryGenerator<B, T>(this->m_builder, ExternalUnaryOperation::Logarithm);
			case BuiltinFunction::Exponential:
				return new ExternalUnaryGenerator<B, T>(this->m_builder, ExternalUnaryOperation::Exponential);
			case BuiltinFunction::Cosine:
				return new ExternalUnaryGenerator<B, T>(this->m_builder, ExternalUnaryOperation::Cosine);
			case BuiltinFunction::Sine:
				return new ExternalUnaryGenerator<B, T>(this->m_builder, ExternalUnaryOperation::Sine);
			case BuiltinFunction::Tangent:
				return new ExternalUnaryGenerator<B, T>(this->m_builder, ExternalUnaryOperation::Tangent);
			case BuiltinFunction::InverseCosine:
				return new ExternalUnaryGenerator<B, T>(this->m_builder, ExternalUnaryOperation::InverseCosine);
			case BuiltinFunction::InverseSine:
				return new ExternalUnaryGenerator<B, T>(this->m_builder, ExternalUnaryOperation::InverseSine);
			case BuiltinFunction::InverseTangent:
				return new ExternalUnaryGenerator<B, T>(this->m_builder, ExternalUnaryOperation::InverseTangent);
			case BuiltinFunction::HyperbolicCosine:
				return new ExternalUnaryGenerator<B, T>(this->m_builder, ExternalUnaryOperation::HyperbolicCosine);
			case BuiltinFunction::HyperbolicSine:
				return new ExternalUnaryGenerator<B, T>(this->m_builder, ExternalUnaryOperation::HyperbolicSine);
			case BuiltinFunction::HyperbolicTangent:
				return new ExternalUnaryGenerator<B, T>(this->m_builder, ExternalUnaryOperation::HyperbolicTangent);
			case BuiltinFunction::HyperbolicInverseCosine:
				return new ExternalUnaryGenerator<B, T>(this->m_builder, ExternalUnaryOperation::HyperbolicInverseCosine);
			case BuiltinFunction::HyperbolicInverseSine:
				return new ExternalUnaryGenerator<B, T>(this->m_builder, ExternalUnaryOperation::HyperbolicInverseSine);
			case BuiltinFunction::HyperbolicInverseTangent:
				return new ExternalUnaryGenerator<B, T>(this->m_builder, ExternalUnaryOperation::HyperbolicInverseTangent);
			case BuiltinFunction::Less:
				return new ComparisonGenerator<B, T>(this->m_builder, ComparisonOperator::Less);
			case BuiltinFunction::Greater:
				return new ComparisonGenerator<B, T>(this->m_builder, ComparisonOperator::Greater);
			case BuiltinFunction::LessEqual:
				return new ComparisonGenerator<B, T>(this->m_builder, ComparisonOperator::LessEqual);
			case BuiltinFunction::GreaterEqual:
				return new ComparisonGenerator<B, T>(this->m_builder, ComparisonOperator::GreaterEqual);
			case BuiltinFunction::Equal:
				return new ComparisonGenerator<B, T>(this->m_builder, ComparisonOperator::Equal);
			case BuiltinFunction::NotEqual:
				return new ComparisonGenerator<B, T>(this->m_builder, ComparisonOperator::NotEqual);
			case BuiltinFunction::Plus:
				return new BinaryGenerator<B, T>(this->m_builder, BinaryOperation::Plus);
			case BuiltinFunction::Minus:
				return new BinaryGenerator<B, T>(this->m_builder, BinaryOperation::Minus);
			case BuiltinFunction::Multiply:
				return new BinaryGenerator<B, T>(this->m_builder, BinaryOperation::Multiply);
			case BuiltinFunction::Divide:
				return new BinaryGenerator<B, T>(this->m_builder, BinaryOperation::Divide);
			case BuiltinFunction::Power:
				return new ExternalBinaryGenerator<B, T>(this->m_builder, ExternalBinaryOperation::Power);
			case BuiltinFunction::Logarithm2:
				return new ExternalBinaryGenerator<B, T>(this->m_builder, ExternalBinaryOperation::Logarithm);
			case BuiltinFunction::Modulo:
				return new ExternalBinaryGenerator<B, T>(this->m_builder, ExternalBinaryOperation::Modulo);
			case BuiltinFunction::And:
				return new BinaryGenerator<B, T>(this->m_builder, BinaryOperation::And);
			case BuiltinFunction::Or:
				return new BinaryGenerator<B, T>(this->m_builder, BinaryOperation::Or);
			case BuiltinFunction::Nand:
				return new BinaryGenerator<B, T>(this->m_builder, BinaryOperation::Nand);
			case BuiltinFunction::Nor:
				return new BinaryGenerator<B, T>(this->m_builder, BinaryOperation::Nor);
			case BuiltinFunction::Xor:
				return new BinaryGenerator<B, T>(this->m_builder, BinaryOperation::Xor);
			case BuiltinFunction::Compress:
				return new CompressionGenerator<B, T>(this->m_builder);
			case BuiltinFunction::Count:
				return new ReductionGenerator<B, T>(this->m_builder, ReductionOperation::Count);
			case BuiltinFunction::Sum:
				return new ReductionGenerator<B, T>(this->m_builder, ReductionOperation::Sum);
			case BuiltinFunction::Average:
				return new ReductionGenerator<B, T>(this->m_builder, ReductionOperation::Average);
			case BuiltinFunction::Minimum:
				return new ReductionGenerator<B, T>(this->m_builder, ReductionOperation::Minimum);
			case BuiltinFunction::Maximum:
				return new ReductionGenerator<B, T>(this->m_builder, ReductionOperation::Maximum);
			case BuiltinFunction::Fill:
				return new FillGenerator<B, T>(this->m_builder);
			case BuiltinFunction::Unsupported:
				return nullptr;
		}
	}

protected:
	const std::string& m_target = nullptr;
};

}
