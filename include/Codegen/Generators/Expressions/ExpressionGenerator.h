#pragma once

#include "HorseIR/Traversal/ConstVisitor.h"
#include "Codegen/Generators/Generator.h"

#include "HorseIR/Tree/Tree.h"
#include "HorseIR/Utils/TypeUtils.h"

#include "Codegen/Builder.h"
#include "Codegen/Generators/TypeDispatch.h"
#include "Codegen/Generators/Expressions/Builtins/BuiltinGenerator.h"
#include "Codegen/Generators/Expressions/Builtins/BinaryGenerator.h"
#include "Codegen/Generators/Expressions/Builtins/ComparisonGenerator.h"
#include "Codegen/Generators/Expressions/Builtins/CompressionGenerator.h"
#include "Codegen/Generators/Expressions/Builtins/ExternalBinaryGenerator.h"
#include "Codegen/Generators/Expressions/Builtins/ExternalUnaryGenerator.h"
// #include "Codegen/Generators/Expressions/Builtins/FillGenerator.h"
#include "Codegen/Generators/Expressions/Builtins/ReductionGenerator.h"
#include "Codegen/Generators/Expressions/Builtins/RoundingGenerator.h"
#include "Codegen/Generators/Expressions/Builtins/UnaryGenerator.h"

#include "PTX/PTX.h"

#include "Utils/Logger.h"

namespace Codegen {

template<PTX::Bits B>
class ExpressionGenerator : public HorseIR::ConstVisitor, public Generator
{
public:
	ExpressionGenerator(Builder& builder) : Generator(builder) {}

	void Generate(const std::vector<HorseIR::LValue *>& targets, const HorseIR::Expression *expression)
	{
		m_targets = targets;
		expression->Accept(*this);
	}

	void Generate(const HorseIR::Expression *expression)
	{
		m_targets.clear();
		expression->Accept(*this);
	}

	void Visit(const HorseIR::CallExpression *call) override
	{
		auto function = call->GetFunctionLiteral()->GetFunction();
		switch (function->GetKind())
		{
			case HorseIR::FunctionDeclaration::Kind::Builtin:
			{
				if (m_targets.size() != 1)
				{
					Utils::Logger::LogError("Builtin function generators expect a single target");
				}
				DispatchType(*this, HorseIR::TypeUtils::GetSingleType(call->GetTypes()), static_cast<const HorseIR::BuiltinFunction *>(function), call);
				break;
			}
			case HorseIR::FunctionDeclaration::Kind::Definition:
			{
				Utils::Logger::LogError("Generator for user defined functions not implemented");
			}
		}
	}

	template<class T>
	void Generate(const HorseIR::BuiltinFunction *function, const HorseIR::CallExpression *call)
	{
		BuiltinGenerator<B, T> *generator = GetBuiltinGenerator<T>(function);
		if (generator == nullptr)
		{
			Utils::Logger::LogError("Generator for builtin function " + function->GetName() + " not implemented");
		}
		generator->Generate(m_targets.at(0), call);
		delete generator;
	}

	template<typename T>
	BuiltinGenerator<B, T> *GetBuiltinGenerator(const HorseIR::BuiltinFunction *function)
	{
		switch (function->GetPrimitive())
		{
			// Unary
			case HorseIR::BuiltinFunction::Primitive::Absolute:
				return new UnaryGenerator<B, T>(this->m_builder, UnaryOperation::Absolute);
			case HorseIR::BuiltinFunction::Primitive::Negate:
				return new UnaryGenerator<B, T>(this->m_builder, UnaryOperation::Negate);
			case HorseIR::BuiltinFunction::Primitive::Ceiling:
				return new RoundingGenerator<B, T>(this->m_builder, RoundingOperation::Ceiling);
			case HorseIR::BuiltinFunction::Primitive::Floor:
				return new RoundingGenerator<B, T>(this->m_builder, RoundingOperation::Floor);
			case HorseIR::BuiltinFunction::Primitive::Round:
				return new RoundingGenerator<B, T>(this->m_builder, RoundingOperation::Nearest);
			case HorseIR::BuiltinFunction::Primitive::Reciprocal:
				return new UnaryGenerator<B, T>(this->m_builder, UnaryOperation::Reciprocal);
			case HorseIR::BuiltinFunction::Primitive::Conjugate:
				//TODO: Add support for complex numbers
				Utils::Logger::LogError("Complex number function 'conj' is not supported");
			case HorseIR::BuiltinFunction::Primitive::Sign:
				return new ComparisonGenerator<B, T>(this->m_builder, ComparisonOperator::Sign);
			case HorseIR::BuiltinFunction::Primitive::Pi:
				return new UnaryGenerator<B, T>(this->m_builder, UnaryOperation::Pi);
			case HorseIR::BuiltinFunction::Primitive::Not:
				return new UnaryGenerator<B, T>(this->m_builder, UnaryOperation::Not);
			case HorseIR::BuiltinFunction::Primitive::Logarithm:
				return new ExternalUnaryGenerator<B, T>(this->m_builder, ExternalUnaryOperation::Logarithm);
			case HorseIR::BuiltinFunction::Primitive::Logarithm2:
				return new ExternalUnaryGenerator<B, T>(this->m_builder, ExternalUnaryOperation::Logarithm2);
			case HorseIR::BuiltinFunction::Primitive::Logarithm10:
				return new ExternalUnaryGenerator<B, T>(this->m_builder, ExternalUnaryOperation::Logarithm10);
			case HorseIR::BuiltinFunction::Primitive::SquareRoot:
				return new ExternalUnaryGenerator<B, T>(this->m_builder, ExternalUnaryOperation::SquareRoot);
			case HorseIR::BuiltinFunction::Primitive::Exponential:
				return new ExternalUnaryGenerator<B, T>(this->m_builder, ExternalUnaryOperation::Exponential);
			case HorseIR::BuiltinFunction::Primitive::Cosine:
				return new ExternalUnaryGenerator<B, T>(this->m_builder, ExternalUnaryOperation::Cosine);
			case HorseIR::BuiltinFunction::Primitive::Sine:
				return new ExternalUnaryGenerator<B, T>(this->m_builder, ExternalUnaryOperation::Sine);
			case HorseIR::BuiltinFunction::Primitive::Tangent:
				return new ExternalUnaryGenerator<B, T>(this->m_builder, ExternalUnaryOperation::Tangent);
			case HorseIR::BuiltinFunction::Primitive::InverseCosine:
				return new ExternalUnaryGenerator<B, T>(this->m_builder, ExternalUnaryOperation::InverseCosine);
			case HorseIR::BuiltinFunction::Primitive::InverseSine:
				return new ExternalUnaryGenerator<B, T>(this->m_builder, ExternalUnaryOperation::InverseSine);
			case HorseIR::BuiltinFunction::Primitive::InverseTangent:
				return new ExternalUnaryGenerator<B, T>(this->m_builder, ExternalUnaryOperation::InverseTangent);
			case HorseIR::BuiltinFunction::Primitive::HyperbolicCosine:
				return new ExternalUnaryGenerator<B, T>(this->m_builder, ExternalUnaryOperation::HyperbolicCosine);
			case HorseIR::BuiltinFunction::Primitive::HyperbolicSine:
				return new ExternalUnaryGenerator<B, T>(this->m_builder, ExternalUnaryOperation::HyperbolicSine);
			case HorseIR::BuiltinFunction::Primitive::HyperbolicTangent:
				return new ExternalUnaryGenerator<B, T>(this->m_builder, ExternalUnaryOperation::HyperbolicTangent);
			case HorseIR::BuiltinFunction::Primitive::HyperbolicInverseCosine:
				return new ExternalUnaryGenerator<B, T>(this->m_builder, ExternalUnaryOperation::HyperbolicInverseCosine);
			case HorseIR::BuiltinFunction::Primitive::HyperbolicInverseSine:
				return new ExternalUnaryGenerator<B, T>(this->m_builder, ExternalUnaryOperation::HyperbolicInverseSine);
			case HorseIR::BuiltinFunction::Primitive::HyperbolicInverseTangent:
				return new ExternalUnaryGenerator<B, T>(this->m_builder, ExternalUnaryOperation::HyperbolicInverseTangent);

			// Binary
			case HorseIR::BuiltinFunction::Primitive::Less:
				return new ComparisonGenerator<B, T>(this->m_builder, ComparisonOperator::Less);
			case HorseIR::BuiltinFunction::Primitive::Greater:
				return new ComparisonGenerator<B, T>(this->m_builder, ComparisonOperator::Greater);
			case HorseIR::BuiltinFunction::Primitive::LessEqual:
				return new ComparisonGenerator<B, T>(this->m_builder, ComparisonOperator::LessEqual);
			case HorseIR::BuiltinFunction::Primitive::GreaterEqual:
				return new ComparisonGenerator<B, T>(this->m_builder, ComparisonOperator::GreaterEqual);
			case HorseIR::BuiltinFunction::Primitive::Equal:
				return new ComparisonGenerator<B, T>(this->m_builder, ComparisonOperator::Equal);
			case HorseIR::BuiltinFunction::Primitive::NotEqual:
				return new ComparisonGenerator<B, T>(this->m_builder, ComparisonOperator::NotEqual);
			case HorseIR::BuiltinFunction::Primitive::Plus:
				return new BinaryGenerator<B, T>(this->m_builder, BinaryOperation::Plus);
			case HorseIR::BuiltinFunction::Primitive::Minus:
				return new BinaryGenerator<B, T>(this->m_builder, BinaryOperation::Minus);
			case HorseIR::BuiltinFunction::Primitive::Multiply:
				return new BinaryGenerator<B, T>(this->m_builder, BinaryOperation::Multiply);
			case HorseIR::BuiltinFunction::Primitive::Divide:
				return new BinaryGenerator<B, T>(this->m_builder, BinaryOperation::Divide);
			case HorseIR::BuiltinFunction::Primitive::Power:
				return new ExternalBinaryGenerator<B, T>(this->m_builder, ExternalBinaryOperation::Power);
			case HorseIR::BuiltinFunction::Primitive::LogarithmBase:
				return new ExternalBinaryGenerator<B, T>(this->m_builder, ExternalBinaryOperation::Logarithm);
			case HorseIR::BuiltinFunction::Primitive::Modulo:
				return new ExternalBinaryGenerator<B, T>(this->m_builder, ExternalBinaryOperation::Modulo);
			case HorseIR::BuiltinFunction::Primitive::And:
				return new BinaryGenerator<B, T>(this->m_builder, BinaryOperation::And);
			case HorseIR::BuiltinFunction::Primitive::Or:
				return new BinaryGenerator<B, T>(this->m_builder, BinaryOperation::Or);
			case HorseIR::BuiltinFunction::Primitive::Nand:
				return new BinaryGenerator<B, T>(this->m_builder, BinaryOperation::Nand);
			case HorseIR::BuiltinFunction::Primitive::Nor:
				return new BinaryGenerator<B, T>(this->m_builder, BinaryOperation::Nor);
			case HorseIR::BuiltinFunction::Primitive::Xor:
				return new BinaryGenerator<B, T>(this->m_builder, BinaryOperation::Xor);

			// Algebraic Binary
			case HorseIR::BuiltinFunction::Primitive::Compress:
				return new CompressionGenerator<B, T>(this->m_builder);
			//TODO: Vector
			// case HorseIR::BuiltinFunction::Primitive::Vector:
				// return new FillGenerator<B, T>(this->m_builder);

			// Reduction
			case HorseIR::BuiltinFunction::Primitive::Length:
				return new ReductionGenerator<B, T>(this->m_builder, ReductionOperation::Length);
			case HorseIR::BuiltinFunction::Primitive::Sum:
				return new ReductionGenerator<B, T>(this->m_builder, ReductionOperation::Sum);
			case HorseIR::BuiltinFunction::Primitive::Average:
				return new ReductionGenerator<B, T>(this->m_builder, ReductionOperation::Average);
			case HorseIR::BuiltinFunction::Primitive::Minimum:
				return new ReductionGenerator<B, T>(this->m_builder, ReductionOperation::Minimum);
			case HorseIR::BuiltinFunction::Primitive::Maximum:
				return new ReductionGenerator<B, T>(this->m_builder, ReductionOperation::Maximum);
		}
		return nullptr;
	}

private:
	std::vector<HorseIR::LValue *> m_targets;
};

}
