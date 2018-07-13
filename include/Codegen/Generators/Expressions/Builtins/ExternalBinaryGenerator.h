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

enum class ExternalBinaryOperation {
	Power,
	Modulo
};

static std::string ExternalBinaryOperationString(ExternalBinaryOperation binaryOp)
{
	switch (binaryOp)
	{
		case ExternalBinaryOperation::Power:
			return "pow";
		case ExternalBinaryOperation::Modulo:
			return "mod";
	}
	return "<unknown>";
}

template<PTX::Bits B, class T, typename Enabled = void>
class ExternalBinaryGenerator : public BuiltinGenerator<B, T>
{
public:
	ExternalBinaryGenerator(const PTX::Register<T> *target, Builder *builder, ExternalBinaryOperation binaryOp) : BuiltinGenerator<B, T>(target, builder), m_binaryOp(binaryOp) {}

private:
	ExternalBinaryOperation m_binaryOp;
};

template<PTX::Bits B, PTX::Bits S>
class ExternalBinaryGenerator<B, PTX::FloatType<S>, std::enable_if_t<S == PTX::Bits::Bits32 || S == PTX::Bits::Bits64>> : public BuiltinGenerator<B, PTX::FloatType<S>>
{
public:
	ExternalBinaryGenerator(const PTX::Register<PTX::FloatType<S>> *target, Builder *builder, ExternalBinaryOperation binaryOp) : BuiltinGenerator<B, PTX::FloatType<S>>(target, builder), m_binaryOp(binaryOp) {}

	void Generate(const HorseIR::CallExpression *call) override
	{
		auto block = new PTX::BlockStatement();
		this->m_builder->AddStatement(block);
		this->m_builder->OpenScope(block);

		PTX::ExternalMathFunctions::BinaryFunction<S> *function = GetFunction(m_binaryOp);

		OperandGenerator<B, PTX::BitType<S>> opGen(this->m_builder);
		auto src1 = opGen.GenerateRegister(call->GetArgument(0));
		auto src2 = opGen.GenerateRegister(call->GetArgument(1));

		auto paramDeclaration = new PTX::ParameterDeclaration<PTX::BitType<S>>("$temp", 3);
		this->m_builder->AddStatement(paramDeclaration);

		auto paramIn1 = paramDeclaration->GetVariable("$temp", 0);
		auto paramIn2 = paramDeclaration->GetVariable("$temp", 1);
		auto paramOut = paramDeclaration->GetVariable("$temp", 2);

		auto addressIn1 = new PTX::MemoryAddress<B, PTX::BitType<S>, PTX::ParameterSpace>(paramIn1);
		auto addressIn2 = new PTX::MemoryAddress<B, PTX::BitType<S>, PTX::ParameterSpace>(paramIn2);
		auto addressOut = new PTX::MemoryAddress<B, PTX::BitType<S>, PTX::ParameterSpace>(paramOut);

		this->m_builder->AddStatement(new PTX::StoreInstruction<B, PTX::BitType<S>, PTX::ParameterSpace>(addressIn1, src1));
		this->m_builder->AddStatement(new PTX::StoreInstruction<B, PTX::BitType<S>, PTX::ParameterSpace>(addressIn2, src2));
		this->m_builder->AddStatement(new PTX::CallInstruction<typename PTX::ExternalMathFunctions::BinaryFunction<S>::Signature>(function, paramOut, paramIn1, paramIn2));
		this->m_builder->AddStatement(new PTX::LoadInstruction<B, PTX::BitType<S>, PTX::ParameterSpace>(new PTX::BitRegisterAdapter<PTX::FloatType, S>(this->m_target), addressOut));

		this->m_builder->CloseScope();
	}

private:
	static PTX::ExternalMathFunctions::BinaryFunction<S> *GetFunction(ExternalBinaryOperation binaryOp)
	{
		switch (binaryOp)
		{
			case ExternalBinaryOperation::Power:
				return PTX::ExternalMathFunctions::pow<S>;
			case ExternalBinaryOperation::Modulo:
				return PTX::ExternalMathFunctions::mod<S>;
			default:
				BuiltinGenerator<B, PTX::Float32Type>::Unimplemented("external function " + ExternalBinaryOperationString(binaryOp));
		}
	}

	ExternalBinaryOperation m_binaryOp;
};

}
