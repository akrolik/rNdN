#pragma once

#include "HorseIR/Traversal/ForwardTraversal.h"
#include "Codegen/Generators/Generator.h"

#include "HorseIR/BuiltinFunctions.h"
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
		HorseIR::BuiltinFunction function = HorseIR::GetBuiltinFunction(call->GetName());
		switch (function)
		{
			case HorseIR::BuiltinFunction::Absolute:
				return new UnaryGenerator<B, T>(this->m_builder, UnaryOperation::Absolute);
			case HorseIR::BuiltinFunction::Negate:
				return new UnaryGenerator<B, T>(this->m_builder, UnaryOperation::Negate);
			case HorseIR::BuiltinFunction::Ceiling:
				return new RoundingGenerator<B, T>(this->m_builder, RoundingOperation::Ceiling);
			case HorseIR::BuiltinFunction::Floor:
				return new RoundingGenerator<B, T>(this->m_builder, RoundingOperation::Floor);
			case HorseIR::BuiltinFunction::Round:
				return new RoundingGenerator<B, T>(this->m_builder, RoundingOperation::Nearest);
			case HorseIR::BuiltinFunction::Conjugate:
				//TODO: Add support for complex numbers
				std::cerr << "[ERROR] Complex number functions are not supported" << std::endl;
				std::exit(EXIT_FAILURE);
			case HorseIR::BuiltinFunction::Reciprocal:
				return new UnaryGenerator<B, T>(this->m_builder, UnaryOperation::Reciprocal);
			case HorseIR::BuiltinFunction::Sign:
				return new ComparisonGenerator<B, T>(this->m_builder, ComparisonOperator::Sign);
			case HorseIR::BuiltinFunction::Pi:
				return new UnaryGenerator<B, T>(this->m_builder, UnaryOperation::Pi);
			case HorseIR::BuiltinFunction::Not:
				return new UnaryGenerator<B, T>(this->m_builder, UnaryOperation::Not);
			case HorseIR::BuiltinFunction::Logarithm:
				return new ExternalUnaryGenerator<B, T>(this->m_builder, ExternalUnaryOperation::Logarithm);
			case HorseIR::BuiltinFunction::Exponential:
				return new ExternalUnaryGenerator<B, T>(this->m_builder, ExternalUnaryOperation::Exponential);
			case HorseIR::BuiltinFunction::Cosine:
				return new ExternalUnaryGenerator<B, T>(this->m_builder, ExternalUnaryOperation::Cosine);
			case HorseIR::BuiltinFunction::Sine:
				return new ExternalUnaryGenerator<B, T>(this->m_builder, ExternalUnaryOperation::Sine);
			case HorseIR::BuiltinFunction::Tangent:
				return new ExternalUnaryGenerator<B, T>(this->m_builder, ExternalUnaryOperation::Tangent);
			case HorseIR::BuiltinFunction::InverseCosine:
				return new ExternalUnaryGenerator<B, T>(this->m_builder, ExternalUnaryOperation::InverseCosine);
			case HorseIR::BuiltinFunction::InverseSine:
				return new ExternalUnaryGenerator<B, T>(this->m_builder, ExternalUnaryOperation::InverseSine);
			case HorseIR::BuiltinFunction::InverseTangent:
				return new ExternalUnaryGenerator<B, T>(this->m_builder, ExternalUnaryOperation::InverseTangent);
			case HorseIR::BuiltinFunction::HyperbolicCosine:
				return new ExternalUnaryGenerator<B, T>(this->m_builder, ExternalUnaryOperation::HyperbolicCosine);
			case HorseIR::BuiltinFunction::HyperbolicSine:
				return new ExternalUnaryGenerator<B, T>(this->m_builder, ExternalUnaryOperation::HyperbolicSine);
			case HorseIR::BuiltinFunction::HyperbolicTangent:
				return new ExternalUnaryGenerator<B, T>(this->m_builder, ExternalUnaryOperation::HyperbolicTangent);
			case HorseIR::BuiltinFunction::HyperbolicInverseCosine:
				return new ExternalUnaryGenerator<B, T>(this->m_builder, ExternalUnaryOperation::HyperbolicInverseCosine);
			case HorseIR::BuiltinFunction::HyperbolicInverseSine:
				return new ExternalUnaryGenerator<B, T>(this->m_builder, ExternalUnaryOperation::HyperbolicInverseSine);
			case HorseIR::BuiltinFunction::HyperbolicInverseTangent:
				return new ExternalUnaryGenerator<B, T>(this->m_builder, ExternalUnaryOperation::HyperbolicInverseTangent);
			case HorseIR::BuiltinFunction::Less:
				return new ComparisonGenerator<B, T>(this->m_builder, ComparisonOperator::Less);
			case HorseIR::BuiltinFunction::Greater:
				return new ComparisonGenerator<B, T>(this->m_builder, ComparisonOperator::Greater);
			case HorseIR::BuiltinFunction::LessEqual:
				return new ComparisonGenerator<B, T>(this->m_builder, ComparisonOperator::LessEqual);
			case HorseIR::BuiltinFunction::GreaterEqual:
				return new ComparisonGenerator<B, T>(this->m_builder, ComparisonOperator::GreaterEqual);
			case HorseIR::BuiltinFunction::Equal:
				return new ComparisonGenerator<B, T>(this->m_builder, ComparisonOperator::Equal);
			case HorseIR::BuiltinFunction::NotEqual:
				return new ComparisonGenerator<B, T>(this->m_builder, ComparisonOperator::NotEqual);
			case HorseIR::BuiltinFunction::Plus:
				return new BinaryGenerator<B, T>(this->m_builder, BinaryOperation::Plus);
			case HorseIR::BuiltinFunction::Minus:
				return new BinaryGenerator<B, T>(this->m_builder, BinaryOperation::Minus);
			case HorseIR::BuiltinFunction::Multiply:
				return new BinaryGenerator<B, T>(this->m_builder, BinaryOperation::Multiply);
			case HorseIR::BuiltinFunction::Divide:
				return new BinaryGenerator<B, T>(this->m_builder, BinaryOperation::Divide);
			case HorseIR::BuiltinFunction::Power:
				return new ExternalBinaryGenerator<B, T>(this->m_builder, ExternalBinaryOperation::Power);
			case HorseIR::BuiltinFunction::Logarithm2:
				return new ExternalBinaryGenerator<B, T>(this->m_builder, ExternalBinaryOperation::Logarithm);
			case HorseIR::BuiltinFunction::Modulo:
				return new ExternalBinaryGenerator<B, T>(this->m_builder, ExternalBinaryOperation::Modulo);
			case HorseIR::BuiltinFunction::And:
				return new BinaryGenerator<B, T>(this->m_builder, BinaryOperation::And);
			case HorseIR::BuiltinFunction::Or:
				return new BinaryGenerator<B, T>(this->m_builder, BinaryOperation::Or);
			case HorseIR::BuiltinFunction::Nand:
				return new BinaryGenerator<B, T>(this->m_builder, BinaryOperation::Nand);
			case HorseIR::BuiltinFunction::Nor:
				return new BinaryGenerator<B, T>(this->m_builder, BinaryOperation::Nor);
			case HorseIR::BuiltinFunction::Xor:
				return new BinaryGenerator<B, T>(this->m_builder, BinaryOperation::Xor);
			case HorseIR::BuiltinFunction::Compress:
				return new CompressionGenerator<B, T>(this->m_builder);
			case HorseIR::BuiltinFunction::Count:
				return new ReductionGenerator<B, T>(this->m_builder, ReductionOperation::Count);
			case HorseIR::BuiltinFunction::Sum:
				return new ReductionGenerator<B, T>(this->m_builder, ReductionOperation::Sum);
			case HorseIR::BuiltinFunction::Average:
				return new ReductionGenerator<B, T>(this->m_builder, ReductionOperation::Average);
			case HorseIR::BuiltinFunction::Minimum:
				return new ReductionGenerator<B, T>(this->m_builder, ReductionOperation::Minimum);
			case HorseIR::BuiltinFunction::Maximum:
				return new ReductionGenerator<B, T>(this->m_builder, ReductionOperation::Maximum);
			case HorseIR::BuiltinFunction::Fill:
				return new FillGenerator<B, T>(this->m_builder);
		}
		return nullptr;
	}

protected:
	const std::string& m_target = nullptr;
};

}
