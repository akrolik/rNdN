#pragma once

#include "Codegen/Generators/Expressions/Builtins/BuiltinGenerator.h"

#include "Codegen/Generators/Expressions/Builtins/ExternalUnaryGenerator.h"
#include "Codegen/Generators/Expressions/Builtins/BinaryGenerator.h"

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
	Modulo,
	Logarithm
};

static std::string ExternalBinaryOperationString(ExternalBinaryOperation binaryOp)
{
	switch (binaryOp)
	{
		case ExternalBinaryOperation::Power:
			return "pow";
		case ExternalBinaryOperation::Modulo:
			return "mod";
		case ExternalBinaryOperation::Logarithm:
			return "log2";
	}
	return "<unknown>";
}

template<PTX::Bits B, class T, typename Enabled = void>
class ExternalBinaryGenerator : public BuiltinGenerator<B, T>
{
public:
	ExternalBinaryGenerator(Builder *builder, ExternalBinaryOperation binaryOp) : BuiltinGenerator<B, T>(builder), m_binaryOp(binaryOp) {}

private:
	ExternalBinaryOperation m_binaryOp;
};

template<PTX::Bits B, PTX::Bits S>
class ExternalBinaryGenerator<B, PTX::FloatType<S>, std::enable_if_t<S == PTX::Bits::Bits32 || S == PTX::Bits::Bits64>> : public BuiltinGenerator<B, PTX::FloatType<S>>
{
public:
	ExternalBinaryGenerator(Builder *builder, ExternalBinaryOperation binaryOp) : BuiltinGenerator<B, PTX::FloatType<S>>(builder), m_binaryOp(binaryOp) {}

	void Generate(const PTX::Register<PTX::FloatType<S>> *target, const HorseIR::CallExpression *call) override
	{
		if (m_binaryOp == ExternalBinaryOperation::Logarithm)
		{
			auto block = new PTX::BlockStatement();
			this->m_builder->AddStatement(block);
			auto resources = this->m_builder->OpenScope(block);

			OperandGenerator<B, PTX::BitType<S>> opGen(this->m_builder);
			auto base = opGen.GenerateRegister(call->GetArgument(0));
			auto value = opGen.GenerateRegister(call->GetArgument(1));

			auto temp1 = resources->template AllocateRegister<PTX::FloatType<S>, ResourceKind::Internal>("temp1");
			auto temp2 = resources->template AllocateRegister<PTX::FloatType<S>, ResourceKind::Internal>("temp2");

			ExternalUnaryGenerator<B, PTX::FloatType<S>> gen(this->m_builder, ExternalUnaryOperation::Logarithm);
			gen.Generate(temp1, base);
			gen.Generate(temp2, value);

			BinaryGenerator<B, PTX::FloatType<S>> gendiv(this->m_builder, BinaryOperation::Divide);
			gendiv.Generate(target, temp1, temp2);

			this->m_builder->CloseScope();
		}
		else
		{
			OperandGenerator<B, PTX::BitType<S>> opGen(this->m_builder);
			auto src1 = opGen.GenerateRegister(call->GetArgument(0));
			auto src2 = opGen.GenerateRegister(call->GetArgument(1));
			Generate(target, src1, src2);
		}
	}

	void Generate(const PTX::Register<PTX::FloatType<S>> *target, const PTX::Register<PTX::BitType<S>> *src1, const PTX::Register<PTX::BitType<S>> *src2)
	{
		auto block = new PTX::BlockStatement();
		this->m_builder->AddStatement(block);
		this->m_builder->OpenScope(block);

		PTX::ExternalMathFunctions::BinaryFunction<S> *function = GetFunction(m_binaryOp);
		this->m_builder->AddExternalDeclaration(function);

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
		this->m_builder->AddStatement(new PTX::LoadInstruction<B, PTX::BitType<S>, PTX::ParameterSpace>(new PTX::BitRegisterAdapter<PTX::FloatType, S>(target), addressOut));

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
