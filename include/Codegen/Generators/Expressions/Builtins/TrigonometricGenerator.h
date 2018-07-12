#pragma once

#include "Codegen/Generators/Expressions/Builtins/BuiltinGenerator.h"

#include "Codegen/Generators/TypeDispatch.h"
#include "Codegen/Generators/TypeUtils.h"

#include "PTX/Instructions/DevInstruction.h"
#include "PTX/Instructions/ControlFlow/CallInstruction.h"
#include "PTX/Instructions/Data/LoadInstruction.h"
#include "PTX/Instructions/Data/MoveInstruction.h"
#include "PTX/Instructions/Data/StoreInstruction.h"
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
			return "sinf";
		case TrigonometricOperation::Cosine:
			return "cosf";
		case TrigonometricOperation::Tangent:
			return "tanf";
		case TrigonometricOperation::HyperbolicSine:
			return "sinhf";
		case TrigonometricOperation::HyperbolicCosine:
			return "coshf";
		case TrigonometricOperation::HyperbolicTangent:
			return "tanhf";
		case TrigonometricOperation::InverseSine:
			return "asinf";
		case TrigonometricOperation::InverseCosine:
			return "acosf";
		case TrigonometricOperation::InverseTangent:
			return "atanf";
		case TrigonometricOperation::HyperbolicInverseSine:
			return "asinhf";
		case TrigonometricOperation::HyperbolicInverseCosine:
			return "acoshf";
		case TrigonometricOperation::HyperbolicInverseTangent:
			return "atanhf";
	}
	return "<unknown>";
}
template<PTX::Bits B, class T>
class TrigonometricGenerator : public BuiltinGenerator<B, T>
{
public:
	TrigonometricGenerator(const PTX::Register<T> *target, Builder *builder, TrigonometricOperation trigOp) : BuiltinGenerator<B, T>(target, builder), m_trigOp(trigOp) {}

private:
	TrigonometricOperation m_trigOp;
};

template<PTX::Bits B>
class TrigonometricGenerator<B, PTX::Float32Type> : public BuiltinGenerator<B, PTX::Float32Type>
{
public:
	TrigonometricGenerator(const PTX::Register<PTX::Float32Type> *target, Builder *builder, TrigonometricOperation trigOp) : BuiltinGenerator<B, PTX::Float32Type>(target, builder), m_trigOp(trigOp) {}

	void Generate(const HorseIR::CallExpression *call) override
	{
		std::string functionName = FunctionName(m_trigOp);

		auto function = new PTX::FunctionDefinition<PTX::ParameterVariable<PTX::Float32Type>(PTX::ParameterVariable<PTX::Float32Type>)>();
		function->SetName(functionName);

		OperandGenerator<B, PTX::Float32Type> opGen(this->m_builder);
		auto src = opGen.GenerateOperand(call->GetArgument(0));

		auto block = new PTX::BlockStatement();
		this->m_builder->AddStatement(block);
		this->m_builder->OpenScope(block);

		auto temp = this->m_builder->template AllocateRegister<PTX::Float32Type, ResourceKind::Internal>("temp");

		auto paramDeclaration = new PTX::ParameterDeclaration<PTX::Float32Type>("$temp", 2);
		this->m_builder->AddStatement(paramDeclaration);

		auto paramIn = paramDeclaration->GetVariable("$temp", 0);
		auto paramOut = paramDeclaration->GetVariable("$temp", 1);

		auto addressIn = new PTX::MemoryAddress<B, PTX::Float32Type, PTX::ParameterSpace>(paramIn);
		auto addressOut = new PTX::MemoryAddress<B, PTX::Float32Type, PTX::ParameterSpace>(paramOut);

		this->m_builder->AddStatement(new PTX::MoveInstruction<PTX::Float32Type>(temp, src));
		this->m_builder->AddStatement(new PTX::StoreInstruction<B, PTX::Float32Type, PTX::ParameterSpace>(addressIn, temp));
		this->m_builder->AddStatement(new PTX::CallInstruction<PTX::ParameterVariable<PTX::Float32Type>(PTX::ParameterVariable<PTX::Float32Type>)>(function, paramOut, paramIn));
 		// this->m_builder->AddStatement(new PTX::DevInstruction("call ($temp1), nv_sinf, ($temp0)"));
		this->m_builder->AddStatement(new PTX::LoadInstruction<B, PTX::Float32Type, PTX::ParameterSpace>(this->m_target, addressOut));

		this->m_builder->CloseScope();
	}

private:
	static std::string FunctionName(TrigonometricOperation trigOp)
	{
		switch (trigOp)
		{
			case TrigonometricOperation::Sine:
				return "nv_sinf";
			case TrigonometricOperation::Cosine:
				return "nv_cosf";
			case TrigonometricOperation::Tangent:
				return "nv_tanf";
			case TrigonometricOperation::HyperbolicSine:
				return "nv_sinhf";
			case TrigonometricOperation::HyperbolicCosine:
				return "nv_coshf";
			case TrigonometricOperation::HyperbolicTangent:
				return "nv_tanhf";
			case TrigonometricOperation::InverseSine:
				return "nv_asinf";
			case TrigonometricOperation::InverseCosine:
				return "nv_acosf";
			case TrigonometricOperation::InverseTangent:
				return "nv_atanf";
			case TrigonometricOperation::HyperbolicInverseSine:
				return "nv_asinhf";
			case TrigonometricOperation::HyperbolicInverseCosine:
				return "nv_acoshf";
			case TrigonometricOperation::HyperbolicInverseTangent:
				return "nv_atanhf";
			default:
				BuiltinGenerator<B, PTX::Float32Type>::Unimplemented("trigonometric function " + TrigonometricOperationString(trigOp));
				break;
		}
	}

	TrigonometricOperation m_trigOp;
};

}
