#pragma once

#include "Frontend/Codegen/Generators/Expressions/Builtins/BuiltinGenerator.h"

#include "Frontend/Codegen/Builder.h"
#include "Frontend/Codegen/Generators/Expressions/ConversionGenerator.h"
#include "Frontend/Codegen/Generators/Expressions/OperandCompressionGenerator.h"
#include "Frontend/Codegen/Generators/Expressions/OperandGenerator.h"

#include "HorseIR/Tree/Tree.h"

#include "PTX/Tree/Tree.h"

namespace Frontend {
namespace Codegen {

enum class ExternalUnaryOperation {
	// Trigonometric
	Cosine,
	Sine,
	Tangent,
	InverseCosine,
	InverseSine,
	InverseTangent,
	HyperbolicCosine,
	HyperbolicSine,
	HyperbolicTangent,
	HyperbolicInverseCosine,
	HyperbolicInverseSine,
	HyperbolicInverseTangent,

	// Exponential
	Exponential,
	Logarithm,
	Logarithm2,
	Logarithm10,
	SquareRoot
};

static std::string ExternalUnaryOperationString(ExternalUnaryOperation unaryOp)
{
	switch (unaryOp)
	{
		case ExternalUnaryOperation::Sine:
			return "sin";
		case ExternalUnaryOperation::Cosine:
			return "cos";
		case ExternalUnaryOperation::Tangent:
			return "tan";
		case ExternalUnaryOperation::HyperbolicSine:
			return "sinh";
		case ExternalUnaryOperation::HyperbolicCosine:
			return "cosh";
		case ExternalUnaryOperation::HyperbolicTangent:
			return "tanh";
		case ExternalUnaryOperation::InverseSine:
			return "asin";
		case ExternalUnaryOperation::InverseCosine:
			return "acos";
		case ExternalUnaryOperation::InverseTangent:
			return "atan";
		case ExternalUnaryOperation::HyperbolicInverseSine:
			return "asinh";
		case ExternalUnaryOperation::HyperbolicInverseCosine:
			return "acosh";
		case ExternalUnaryOperation::HyperbolicInverseTangent:
			return "atanh";
		case ExternalUnaryOperation::Exponential:
			return "exp";
		case ExternalUnaryOperation::Logarithm:
			return "log";
		case ExternalUnaryOperation::Logarithm2:
			return "log2";
		case ExternalUnaryOperation::Logarithm10:
			return "log10";
		case ExternalUnaryOperation::SquareRoot:
			return "sqrt";
	}
	return "<unknown>";
}

template<PTX::Bits B, class T, typename Enabled = void>
class ExternalUnaryGenerator : public BuiltinGenerator<B, T>
{
public:
	ExternalUnaryGenerator(Builder& builder, ExternalUnaryOperation unaryOp) : BuiltinGenerator<B, T>(builder), m_unaryOp(unaryOp) {}

	std::string Name() const override { return "ExternalUnaryGenerator"; }

private:
	ExternalUnaryOperation m_unaryOp;
};

template<PTX::Bits B, PTX::Bits S>
class ExternalUnaryGenerator<B, PTX::FloatType<S>, std::enable_if_t<S == PTX::Bits::Bits32 || S == PTX::Bits::Bits64>> : public BuiltinGenerator<B, PTX::FloatType<S>>
{
public:
	ExternalUnaryGenerator(Builder& builder, ExternalUnaryOperation unaryOp) : BuiltinGenerator<B, PTX::FloatType<S>>(builder), m_unaryOp(unaryOp) {}

	std::string Name() const override { return "ExternalUnaryGenerator"; }

	PTX::Register<PTX::PredicateType> *GenerateCompressionPredicate(const std::vector<const HorseIR::Operand *>& arguments) override
	{
		return OperandCompressionGenerator::UnaryCompressionRegister(this->m_builder, arguments);
	}

	PTX::Register<PTX::FloatType<S>> *Generate(const HorseIR::LValue *target, const std::vector<const HorseIR::Operand *>& arguments) override
	{
		OperandGenerator<B, PTX::FloatType<S>> opGen(this->m_builder);
		auto src = opGen.GenerateRegister(arguments.at(0), OperandGenerator<B, PTX::FloatType<S>>::LoadKind::Vector);

		auto targetRegister = this->GenerateTargetRegister(target, arguments);
		Generate(targetRegister, src);

		return targetRegister;
	}
	
	void Generate(PTX::Register<PTX::FloatType<S>> *target, PTX::Register<PTX::FloatType<S>> *src)
	{
		auto block = new PTX::BlockStatement();
		this->m_builder.AddStatement(block);
		this->m_builder.OpenScope(block);

		PTX::ExternalMathFunctions::UnaryFunction<S> *function = GetFunction(m_unaryOp);

		auto globalResources = this->m_builder.GetGlobalResources();
		globalResources->AddExternalFunction(function);

		auto paramDeclaration = new PTX::ParameterDeclaration<PTX::BitType<S>>("$temp", 2);
		this->m_builder.AddStatement(new PTX::DeclarationStatement(paramDeclaration));

		auto paramIn = paramDeclaration->GetVariable("$temp", 0);
		auto paramOut = paramDeclaration->GetVariable("$temp", 1);

		auto addressIn = new PTX::MemoryAddress<B, PTX::BitType<S>, PTX::ParameterSpace>(paramIn);
		auto addressOut = new PTX::MemoryAddress<B, PTX::BitType<S>, PTX::ParameterSpace>(paramOut);

		auto converted = ConversionGenerator::ConvertSource<PTX::BitType<S>, PTX::FloatType<S>>(this->m_builder, src);

		this->m_builder.AddStatement(new PTX::StoreInstruction<B, PTX::BitType<S>, PTX::ParameterSpace>(addressIn, converted));
		this->m_builder.AddStatement(new PTX::CallInstruction<typename PTX::ExternalMathFunctions::UnaryFunction<S>::Signature>(function, paramOut, paramIn));
		this->m_builder.AddStatement(new PTX::LoadInstruction<B, PTX::BitType<S>, PTX::ParameterSpace>(new PTX::BitRegisterAdapter<PTX::FloatType, S>(target), addressOut));

		this->m_builder.CloseScope();
	}

private:
	PTX::ExternalMathFunctions::UnaryFunction<S> *GetFunction(ExternalUnaryOperation unaryOp) const
	{
		switch (unaryOp)
		{
			case ExternalUnaryOperation::Cosine:
				return PTX::ExternalMathFunctions::cos<S>;
			case ExternalUnaryOperation::Sine:
				return PTX::ExternalMathFunctions::sin<S>;
			case ExternalUnaryOperation::Tangent:
				return PTX::ExternalMathFunctions::tan<S>;
			case ExternalUnaryOperation::InverseCosine:
				return PTX::ExternalMathFunctions::acos<S>;
			case ExternalUnaryOperation::InverseSine:
				return PTX::ExternalMathFunctions::asin<S>;
			case ExternalUnaryOperation::InverseTangent:
				return PTX::ExternalMathFunctions::atan<S>;
			case ExternalUnaryOperation::HyperbolicCosine:
				return PTX::ExternalMathFunctions::cosh<S>;
			case ExternalUnaryOperation::HyperbolicSine:
				return PTX::ExternalMathFunctions::sinh<S>;
			case ExternalUnaryOperation::HyperbolicTangent:
				return PTX::ExternalMathFunctions::tanh<S>;
			case ExternalUnaryOperation::HyperbolicInverseCosine:
				return PTX::ExternalMathFunctions::acosh<S>;
			case ExternalUnaryOperation::HyperbolicInverseSine:
				return PTX::ExternalMathFunctions::asinh<S>;
			case ExternalUnaryOperation::HyperbolicInverseTangent:
				return PTX::ExternalMathFunctions::atanh<S>;
			case ExternalUnaryOperation::Exponential:
				return PTX::ExternalMathFunctions::exp<S>;
			case ExternalUnaryOperation::Logarithm:
				return PTX::ExternalMathFunctions::log<S>;
			case ExternalUnaryOperation::Logarithm2:
				return PTX::ExternalMathFunctions::log2<S>;
			case ExternalUnaryOperation::Logarithm10:
				return PTX::ExternalMathFunctions::log10<S>;
			case ExternalUnaryOperation::SquareRoot:
				return PTX::ExternalMathFunctions::sqrt<S>;
			default:
				BuiltinGenerator<B, PTX::FloatType<S>>::Unimplemented("external function " + ExternalUnaryOperationString(unaryOp));
		}
	}

	ExternalUnaryOperation m_unaryOp;
};

}
}
