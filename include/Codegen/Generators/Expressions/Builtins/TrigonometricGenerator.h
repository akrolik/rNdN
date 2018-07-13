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

enum class TrigonometricOperation {
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
	HyperbolicInverseTangent
};

static std::string TrigonometricOperationString(TrigonometricOperation trigOp)
{
	switch (trigOp)
	{
		case TrigonometricOperation::Sine:
			return "sin";
		case TrigonometricOperation::Cosine:
			return "cos";
		case TrigonometricOperation::Tangent:
			return "tan";
		case TrigonometricOperation::HyperbolicSine:
			return "sinh";
		case TrigonometricOperation::HyperbolicCosine:
			return "cosh";
		case TrigonometricOperation::HyperbolicTangent:
			return "tanh";
		case TrigonometricOperation::InverseSine:
			return "asin";
		case TrigonometricOperation::InverseCosine:
			return "acos";
		case TrigonometricOperation::InverseTangent:
			return "atan";
		case TrigonometricOperation::HyperbolicInverseSine:
			return "asinh";
		case TrigonometricOperation::HyperbolicInverseCosine:
			return "acosh";
		case TrigonometricOperation::HyperbolicInverseTangent:
			return "atanh";
	}
	return "<unknown>";
}

template<PTX::Bits B, class T, typename Enabled = void>
class TrigonometricGenerator : public BuiltinGenerator<B, T>
{
public:
	TrigonometricGenerator(const PTX::Register<T> *target, Builder *builder, TrigonometricOperation trigOp) : BuiltinGenerator<B, T>(target, builder), m_trigOp(trigOp) {}

private:
	TrigonometricOperation m_trigOp;
};

template<PTX::Bits B, PTX::Bits S>
class TrigonometricGenerator<B, PTX::FloatType<S>, std::enable_if_t<S == PTX::Bits::Bits32 || S == PTX::Bits::Bits64>> : public BuiltinGenerator<B, PTX::FloatType<S>>
{
public:
	TrigonometricGenerator(const PTX::Register<PTX::FloatType<S>> *target, Builder *builder, TrigonometricOperation trigOp) : BuiltinGenerator<B, PTX::FloatType<S>>(target, builder), m_trigOp(trigOp) {}

	void Generate(const HorseIR::CallExpression *call) override
	{
		auto block = new PTX::BlockStatement();
		this->m_builder->AddStatement(block);
		this->m_builder->OpenScope(block);

		PTX::ExternalMathFunction<S> *function = GetFunction(m_trigOp);

		OperandGenerator<B, PTX::BitType<S>> opGen(this->m_builder);
		auto src = opGen.GenerateRegister(call->GetArgument(0));

		auto paramDeclaration = new PTX::ParameterDeclaration<PTX::BitType<S>>("$temp", 2);
		this->m_builder->AddStatement(paramDeclaration);

		auto paramIn = paramDeclaration->GetVariable("$temp", 0);
		auto paramOut = paramDeclaration->GetVariable("$temp", 1);

		auto addressIn = new PTX::MemoryAddress<B, PTX::BitType<S>, PTX::ParameterSpace>(paramIn);
		auto addressOut = new PTX::MemoryAddress<B, PTX::BitType<S>, PTX::ParameterSpace>(paramOut);

		this->m_builder->AddStatement(new PTX::StoreInstruction<B, PTX::BitType<S>, PTX::ParameterSpace>(addressIn, src));
		this->m_builder->AddStatement(new PTX::CallInstruction<typename PTX::ExternalMathFunction<S>::Signature>(function, paramOut, paramIn));
		this->m_builder->AddStatement(new PTX::LoadInstruction<B, PTX::BitType<S>, PTX::ParameterSpace>(new PTX::BitRegisterAdapter<PTX::FloatType, S>(this->m_target), addressOut));

		this->m_builder->CloseScope();
	}

private:
	static PTX::ExternalMathFunction<S> *GetFunction(TrigonometricOperation trigOp)
	{
		switch (trigOp)
		{
			case TrigonometricOperation::Cosine:
				return PTX::ExternalMathFunction_cos<S>;
			case TrigonometricOperation::Sine:
				return PTX::ExternalMathFunction_sin<S>;
			case TrigonometricOperation::Tangent:
				return PTX::ExternalMathFunction_tan<S>;
			case TrigonometricOperation::InverseCosine:
				return PTX::ExternalMathFunction_acos<S>;
			case TrigonometricOperation::InverseSine:
				return PTX::ExternalMathFunction_asin<S>;
			case TrigonometricOperation::InverseTangent:
				return PTX::ExternalMathFunction_atan<S>;
			case TrigonometricOperation::HyperbolicCosine:
				return PTX::ExternalMathFunction_cosh<S>;
			case TrigonometricOperation::HyperbolicSine:
				return PTX::ExternalMathFunction_sinh<S>;
			case TrigonometricOperation::HyperbolicTangent:
				return PTX::ExternalMathFunction_tanh<S>;
			case TrigonometricOperation::HyperbolicInverseCosine:
				return PTX::ExternalMathFunction_acosh<S>;
			case TrigonometricOperation::HyperbolicInverseSine:
				return PTX::ExternalMathFunction_asinh<S>;
			case TrigonometricOperation::HyperbolicInverseTangent:
				return PTX::ExternalMathFunction_atanh<S>;
			default:
				BuiltinGenerator<B, PTX::Float32Type>::Unimplemented("trigonometric function " + TrigonometricOperationString(trigOp));
		}
	}

	TrigonometricOperation m_trigOp;
};

}
