#pragma once

#include "HorseIR/Traversal/ForwardTraversal.h"
#include "Codegen/Generators/Generator.h"

#include "HorseIR/Tree/BuiltinMethod.h"
#include "HorseIR/Tree/Expressions/CallExpression.h"

#include "PTX/Operands/Variables/Register.h"

#include "Codegen/Builder.h"
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

#include "Utils/Logger.h"

namespace Codegen {

template<PTX::Bits B, class T>
class ExpressionGenerator : public HorseIR::ForwardTraversal, public Generator
{
public:
	ExpressionGenerator(const std::string& target, Builder& builder) : Generator(builder), m_target(target) {}

	void Visit(HorseIR::CallExpression *call) override
	{
		auto method = call->GetMethod();
		switch (method->GetKind())
		{
			case HorseIR::MethodDeclaration::Kind::Builtin:
			{
				BuiltinGenerator<B, T> *generator = GetBuiltinGenerator(static_cast<HorseIR::BuiltinMethod *>(method));
				if (generator == nullptr)
				{
					Utils::Logger::LogError("Generator for builtin function " + method->GetName() + " not implemented");
				}
				generator->Generate(m_target, call);
				delete generator;
				break;
			}
			case HorseIR::MethodDeclaration::Kind::Definition:
				Utils::Logger::LogError("Generator for user defined functions not implemented");
				break;
		}

	}

	BuiltinGenerator<B, T> *GetBuiltinGenerator(HorseIR::BuiltinMethod *method)
	{
		switch (method->GetKind())
		{
			case HorseIR::BuiltinMethod::Kind::Absolute:
				return new UnaryGenerator<B, T>(this->m_builder, UnaryOperation::Absolute);
			case HorseIR::BuiltinMethod::Kind::Negate:
				return new UnaryGenerator<B, T>(this->m_builder, UnaryOperation::Negate);
			case HorseIR::BuiltinMethod::Kind::Ceiling:
				return new RoundingGenerator<B, T>(this->m_builder, RoundingOperation::Ceiling);
			case HorseIR::BuiltinMethod::Kind::Floor:
				return new RoundingGenerator<B, T>(this->m_builder, RoundingOperation::Floor);
			case HorseIR::BuiltinMethod::Kind::Round:
				return new RoundingGenerator<B, T>(this->m_builder, RoundingOperation::Nearest);
			case HorseIR::BuiltinMethod::Kind::Reciprocal:
				return new UnaryGenerator<B, T>(this->m_builder, UnaryOperation::Reciprocal);
			case HorseIR::BuiltinMethod::Kind::Conjugate:
				//TODO: Add support for complex numbers
				Utils::Logger::LogError("Complex number function 'conj' is not supported");
			case HorseIR::BuiltinMethod::Kind::Sign:
				return new ComparisonGenerator<B, T>(this->m_builder, ComparisonOperator::Sign);
			case HorseIR::BuiltinMethod::Kind::Pi:
				return new UnaryGenerator<B, T>(this->m_builder, UnaryOperation::Pi);
			case HorseIR::BuiltinMethod::Kind::Not:
				return new UnaryGenerator<B, T>(this->m_builder, UnaryOperation::Not);
			case HorseIR::BuiltinMethod::Kind::Logarithm:
				return new ExternalUnaryGenerator<B, T>(this->m_builder, ExternalUnaryOperation::Logarithm);
			case HorseIR::BuiltinMethod::Kind::Logarithm2:
				return new ExternalUnaryGenerator<B, T>(this->m_builder, ExternalUnaryOperation::Logarithm2);
			case HorseIR::BuiltinMethod::Kind::Logarithm10:
				return new ExternalUnaryGenerator<B, T>(this->m_builder, ExternalUnaryOperation::Logarithm10);
			case HorseIR::BuiltinMethod::Kind::SquareRoot:
				return new ExternalUnaryGenerator<B, T>(this->m_builder, ExternalUnaryOperation::SquareRoot);
			case HorseIR::BuiltinMethod::Kind::Exponential:
				return new ExternalUnaryGenerator<B, T>(this->m_builder, ExternalUnaryOperation::Exponential);
			case HorseIR::BuiltinMethod::Kind::Cosine:
				return new ExternalUnaryGenerator<B, T>(this->m_builder, ExternalUnaryOperation::Cosine);
			case HorseIR::BuiltinMethod::Kind::Sine:
				return new ExternalUnaryGenerator<B, T>(this->m_builder, ExternalUnaryOperation::Sine);
			case HorseIR::BuiltinMethod::Kind::Tangent:
				return new ExternalUnaryGenerator<B, T>(this->m_builder, ExternalUnaryOperation::Tangent);
			case HorseIR::BuiltinMethod::Kind::InverseCosine:
				return new ExternalUnaryGenerator<B, T>(this->m_builder, ExternalUnaryOperation::InverseCosine);
			case HorseIR::BuiltinMethod::Kind::InverseSine:
				return new ExternalUnaryGenerator<B, T>(this->m_builder, ExternalUnaryOperation::InverseSine);
			case HorseIR::BuiltinMethod::Kind::InverseTangent:
				return new ExternalUnaryGenerator<B, T>(this->m_builder, ExternalUnaryOperation::InverseTangent);
			case HorseIR::BuiltinMethod::Kind::HyperbolicCosine:
				return new ExternalUnaryGenerator<B, T>(this->m_builder, ExternalUnaryOperation::HyperbolicCosine);
			case HorseIR::BuiltinMethod::Kind::HyperbolicSine:
				return new ExternalUnaryGenerator<B, T>(this->m_builder, ExternalUnaryOperation::HyperbolicSine);
			case HorseIR::BuiltinMethod::Kind::HyperbolicTangent:
				return new ExternalUnaryGenerator<B, T>(this->m_builder, ExternalUnaryOperation::HyperbolicTangent);
			case HorseIR::BuiltinMethod::Kind::HyperbolicInverseCosine:
				return new ExternalUnaryGenerator<B, T>(this->m_builder, ExternalUnaryOperation::HyperbolicInverseCosine);
			case HorseIR::BuiltinMethod::Kind::HyperbolicInverseSine:
				return new ExternalUnaryGenerator<B, T>(this->m_builder, ExternalUnaryOperation::HyperbolicInverseSine);
			case HorseIR::BuiltinMethod::Kind::HyperbolicInverseTangent:
				return new ExternalUnaryGenerator<B, T>(this->m_builder, ExternalUnaryOperation::HyperbolicInverseTangent);
			case HorseIR::BuiltinMethod::Kind::Less:
				return new ComparisonGenerator<B, T>(this->m_builder, ComparisonOperator::Less);
			case HorseIR::BuiltinMethod::Kind::Greater:
				return new ComparisonGenerator<B, T>(this->m_builder, ComparisonOperator::Greater);
			case HorseIR::BuiltinMethod::Kind::LessEqual:
				return new ComparisonGenerator<B, T>(this->m_builder, ComparisonOperator::LessEqual);
			case HorseIR::BuiltinMethod::Kind::GreaterEqual:
				return new ComparisonGenerator<B, T>(this->m_builder, ComparisonOperator::GreaterEqual);
			case HorseIR::BuiltinMethod::Kind::Equal:
				return new ComparisonGenerator<B, T>(this->m_builder, ComparisonOperator::Equal);
			case HorseIR::BuiltinMethod::Kind::NotEqual:
				return new ComparisonGenerator<B, T>(this->m_builder, ComparisonOperator::NotEqual);
			case HorseIR::BuiltinMethod::Kind::Plus:
				return new BinaryGenerator<B, T>(this->m_builder, BinaryOperation::Plus);
			case HorseIR::BuiltinMethod::Kind::Minus:
				return new BinaryGenerator<B, T>(this->m_builder, BinaryOperation::Minus);
			case HorseIR::BuiltinMethod::Kind::Multiply:
				return new BinaryGenerator<B, T>(this->m_builder, BinaryOperation::Multiply);
			case HorseIR::BuiltinMethod::Kind::Divide:
				return new BinaryGenerator<B, T>(this->m_builder, BinaryOperation::Divide);
			case HorseIR::BuiltinMethod::Kind::Power:
				return new ExternalBinaryGenerator<B, T>(this->m_builder, ExternalBinaryOperation::Power);
			case HorseIR::BuiltinMethod::Kind::LogarithmBase:
				return new ExternalBinaryGenerator<B, T>(this->m_builder, ExternalBinaryOperation::Logarithm);
			case HorseIR::BuiltinMethod::Kind::Modulo:
				return new ExternalBinaryGenerator<B, T>(this->m_builder, ExternalBinaryOperation::Modulo);
			case HorseIR::BuiltinMethod::Kind::And:
				return new BinaryGenerator<B, T>(this->m_builder, BinaryOperation::And);
			case HorseIR::BuiltinMethod::Kind::Or:
				return new BinaryGenerator<B, T>(this->m_builder, BinaryOperation::Or);
			case HorseIR::BuiltinMethod::Kind::Nand:
				return new BinaryGenerator<B, T>(this->m_builder, BinaryOperation::Nand);
			case HorseIR::BuiltinMethod::Kind::Nor:
				return new BinaryGenerator<B, T>(this->m_builder, BinaryOperation::Nor);
			case HorseIR::BuiltinMethod::Kind::Xor:
				return new BinaryGenerator<B, T>(this->m_builder, BinaryOperation::Xor);
			case HorseIR::BuiltinMethod::Kind::Compress:
				return new CompressionGenerator<B, T>(this->m_builder);
			case HorseIR::BuiltinMethod::Kind::Count:
				return new ReductionGenerator<B, T>(this->m_builder, ReductionOperation::Count);
			case HorseIR::BuiltinMethod::Kind::Sum:
				return new ReductionGenerator<B, T>(this->m_builder, ReductionOperation::Sum);
			case HorseIR::BuiltinMethod::Kind::Average:
				return new ReductionGenerator<B, T>(this->m_builder, ReductionOperation::Average);
			case HorseIR::BuiltinMethod::Kind::Minimum:
				return new ReductionGenerator<B, T>(this->m_builder, ReductionOperation::Minimum);
			case HorseIR::BuiltinMethod::Kind::Maximum:
				return new ReductionGenerator<B, T>(this->m_builder, ReductionOperation::Maximum);
			case HorseIR::BuiltinMethod::Kind::Fill:
				return new FillGenerator<B, T>(this->m_builder);
		}
		return nullptr;
	}

protected:
	const std::string& m_target = nullptr;
};

}
