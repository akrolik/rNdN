#pragma once

#include "Codegen/Generators/Expressions/Builtins/BuiltinGenerator.h"

#include "PTX/Functions/ExternalMathFunctions.h"
#include "PTX/Instructions/ControlFlow/CallInstruction.h"
#include "PTX/Instructions/Data/LoadInstruction.h"
#include "PTX/Instructions/Data/StoreInstruction.h"
#include "PTX/Operands/Adapters/BitAdapter.h"
#include "PTX/Operands/Address/MemoryAddress.h"
#include "PTX/Operands/Variables/AddressableVariable.h"

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
	}
	return "<unknown>";
}

template<PTX::Bits B, class T, typename Enabled = void>
class ExternalUnaryGenerator : public BuiltinGenerator<B, T>
{
public:
	ExternalUnaryGenerator(Builder *builder, ExternalUnaryOperation unaryOp) : BuiltinGenerator<B, T>(builder), m_unaryOp(unaryOp) {}

private:
	ExternalUnaryOperation m_unaryOp;
};

template<PTX::Bits B, PTX::Bits S>
class ExternalUnaryGenerator<B, PTX::FloatType<S>, std::enable_if_t<S == PTX::Bits::Bits32 || S == PTX::Bits::Bits64>> : public BuiltinGenerator<B, PTX::FloatType<S>>
{
public:
	ExternalUnaryGenerator(Builder *builder, ExternalUnaryOperation unaryOp) : BuiltinGenerator<B, PTX::FloatType<S>>(builder), m_unaryOp(unaryOp) {}

	void Generate(const PTX::Register<PTX::FloatType<S>> *target, const HorseIR::CallExpression *call) override
	{
		OperandGenerator<B, PTX::BitType<S>> opGen(this->m_builder);
		auto src = opGen.GenerateRegister(call->GetArgument(0));
		Generate(target, src);
	}
	
	void Generate(const PTX::Register<PTX::FloatType<S>> *target, const PTX::Register<PTX::BitType<S>> *src)
	{
		auto block = new PTX::BlockStatement();
		this->m_builder->AddStatement(block);
		this->m_builder->OpenScope(block);

		PTX::ExternalMathFunctions::UnaryFunction<S> *function = GetFunction(m_unaryOp);
		this->m_builder->AddExternalDeclaration(function);

		auto paramDeclaration = new PTX::ParameterDeclaration<PTX::BitType<S>>("$temp", 2);
		this->m_builder->AddStatement(paramDeclaration);

		auto paramIn = paramDeclaration->GetVariable("$temp", 0);
		auto paramOut = paramDeclaration->GetVariable("$temp", 1);

		auto addressIn = new PTX::MemoryAddress<B, PTX::BitType<S>, PTX::ParameterSpace>(paramIn);
		auto addressOut = new PTX::MemoryAddress<B, PTX::BitType<S>, PTX::ParameterSpace>(paramOut);

		this->m_builder->AddStatement(new PTX::StoreInstruction<B, PTX::BitType<S>, PTX::ParameterSpace>(addressIn, src));
		this->m_builder->AddStatement(new PTX::CallInstruction<typename PTX::ExternalMathFunctions::UnaryFunction<S>::Signature>(function, paramOut, paramIn));
		this->m_builder->AddStatement(new PTX::LoadInstruction<B, PTX::BitType<S>, PTX::ParameterSpace>(new PTX::BitRegisterAdapter<PTX::FloatType, S>(target), addressOut));

		this->m_builder->CloseScope();
	}

private:
	static PTX::ExternalMathFunctions::UnaryFunction<S> *GetFunction(ExternalUnaryOperation unaryOp)
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
			default:
				BuiltinGenerator<B, PTX::Float32Type>::Unimplemented("external function " + ExternalUnaryOperationString(unaryOp));
		}
	}

	ExternalUnaryOperation m_unaryOp;
};

}
