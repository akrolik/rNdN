#pragma once

#include "Codegen/Generators/Expressions/Builtins/BuiltinGenerator.h"

#include "Codegen/Builder.h"
#include "Codegen/Generators/Expressions/ConversionGenerator.h"
#include "Codegen/Generators/Expressions/OperandCompressionGenerator.h"
#include "Codegen/Generators/Expressions/OperandGenerator.h"
#include "Codegen/Generators/Expressions/Builtins/ExternalUnaryGenerator.h"
#include "Codegen/Generators/Expressions/Builtins/BinaryGenerator.h"

#include "HorseIR/Tree/Tree.h"

#include "PTX/PTX.h"

namespace Codegen {

enum class ExternalBinaryOperation {
	Power,
	Modulo, // int or float
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
			return "logb";
	}
	return "<unknown>";
}

template<PTX::Bits B, class T, typename Enabled = void>
class ExternalBinaryGenerator : public BuiltinGenerator<B, T>
{
public:
	ExternalBinaryGenerator(Builder& builder, ExternalBinaryOperation binaryOp) : BuiltinGenerator<B, T>(builder), m_binaryOp(binaryOp) {}

	std::string Name() const override { return "ExternalBinaryGenerator"; }

private:
	ExternalBinaryOperation m_binaryOp;
};

template<PTX::Bits B, PTX::Bits S>
class ExternalBinaryGenerator<B, PTX::IntType<S>> : public BuiltinGenerator<B, PTX::IntType<S>>
{
public:
	ExternalBinaryGenerator(Builder& builder, ExternalBinaryOperation binaryOp) : BuiltinGenerator<B, PTX::IntType<S>>(builder), m_binaryOp(binaryOp) {}

	std::string Name() const override { return "ExternalBinaryGenerator"; }

	const PTX::Register<PTX::PredicateType> *GenerateCompressionPredicate(const std::vector<HorseIR::Operand *>& arguments) override
	{
		return OperandCompressionGenerator::BinaryCompressionRegister(this->m_builder, arguments);
	}

	const PTX::Register<PTX::IntType<S>> *Generate(const HorseIR::LValue *target, const std::vector<HorseIR::Operand *>& arguments) override
	{
		if (m_binaryOp != ExternalBinaryOperation::Modulo)
		{
			BuiltinGenerator<B, PTX::IntType<S>>::Unimplemented("external binary operation " + ExternalBinaryOperationString(m_binaryOp));
		}

		auto targetRegister = this->GenerateTargetRegister(target, arguments);
		if constexpr(std::is_same<PTX::IntType<S>, PTX::Int8Type>::value)
		{
			auto resources = this->m_builder.GetLocalResources();
			auto targetRegister16 = resources->template AllocateTemporary<PTX::Int16Type>();

			OperandGenerator<B, PTX::Int16Type> opGen(this->m_builder);
			auto src1 = opGen.GenerateRegister(arguments.at(0), OperandGenerator<B, PTX::Int16Type>::LoadKind::Vector);
			auto src2 = opGen.GenerateRegister(arguments.at(1), OperandGenerator<B, PTX::Int16Type>::LoadKind::Vector);
			this->m_builder.AddStatement(new PTX::RemainderInstruction<PTX::Int16Type>(targetRegister16, src1, src2));

			ConversionGenerator::ConvertSource<PTX::Int8Type, PTX::Int16Type>(this->m_builder, targetRegister, targetRegister16);
		}
		else
		{
			OperandGenerator<B, PTX::IntType<S>> opGen(this->m_builder);
			auto src1 = opGen.GenerateRegister(arguments.at(0), OperandGenerator<B, PTX::IntType<S>>::LoadKind::Vector);
			auto src2 = opGen.GenerateRegister(arguments.at(1), OperandGenerator<B, PTX::IntType<S>>::LoadKind::Vector);
			this->m_builder.AddStatement(new PTX::RemainderInstruction<PTX::IntType<S>>(targetRegister, src1, src2));
		}
		return targetRegister;
	}

private:
	ExternalBinaryOperation m_binaryOp;
};

template<PTX::Bits B, PTX::Bits S>
class ExternalBinaryGenerator<B, PTX::FloatType<S>, std::enable_if_t<S == PTX::Bits::Bits32 || S == PTX::Bits::Bits64>> : public BuiltinGenerator<B, PTX::FloatType<S>>
{
public:
	ExternalBinaryGenerator(Builder& builder, ExternalBinaryOperation binaryOp) : BuiltinGenerator<B, PTX::FloatType<S>>(builder), m_binaryOp(binaryOp) {}

	std::string Name() const override { return "ExternalBinaryGenerator"; }

	const PTX::Register<PTX::PredicateType> *GenerateCompressionPredicate(const std::vector<HorseIR::Operand *>& arguments) override
	{
		return OperandCompressionGenerator::BinaryCompressionRegister(this->m_builder, arguments);
	}

	const PTX::Register<PTX::FloatType<S>> *Generate(const HorseIR::LValue *target, const std::vector<HorseIR::Operand *>& arguments) override
	{
		OperandGenerator<B, PTX::FloatType<S>> opGen(this->m_builder);
		auto src1 = opGen.GenerateRegister(arguments.at(0), OperandGenerator<B, PTX::FloatType<S>>::LoadKind::Vector);
		auto src2 = opGen.GenerateRegister(arguments.at(1), OperandGenerator<B, PTX::FloatType<S>>::LoadKind::Vector);

		auto targetRegister = this->GenerateTargetRegister(target, arguments);
		Generate(targetRegister, src1, src2);

		return targetRegister;
	}

	void Generate(const PTX::Register<PTX::FloatType<S>> *target, const PTX::Register<PTX::FloatType<S>> *src1, const PTX::Register<PTX::FloatType<S>> *src2)
	{
		auto block = new PTX::BlockStatement();
		this->m_builder.AddStatement(block);
		auto resources = this->m_builder.OpenScope(block);

		if (m_binaryOp == ExternalBinaryOperation::Logarithm)
		{
			auto temp1 = resources->template AllocateTemporary<PTX::FloatType<S>>();
			auto temp2 = resources->template AllocateTemporary<PTX::FloatType<S>>();

			ExternalUnaryGenerator<B, PTX::FloatType<S>> gen(this->m_builder, ExternalUnaryOperation::Logarithm);
			gen.Generate(temp1, src1);
			gen.Generate(temp2, src2);

			BinaryGenerator<B, PTX::FloatType<S>> gendiv(this->m_builder, BinaryOperation::Divide);
			gendiv.Generate(target, temp1, temp2);
		}
		else
		{
			PTX::ExternalMathFunctions::BinaryFunction<S> *function = GetFunction(m_binaryOp);

			auto globalResources = this->m_builder.GetGlobalResources();
			globalResources->AddExternalFunction(function);

			auto paramDeclaration = new PTX::ParameterDeclaration<PTX::BitType<S>>("$temp", 3);
			this->m_builder.AddStatement(paramDeclaration);
			this->m_builder.AddStatement(new PTX::BlankStatement());

			auto paramIn1 = paramDeclaration->GetVariable("$temp", 0);
			auto paramIn2 = paramDeclaration->GetVariable("$temp", 1);
			auto paramOut = paramDeclaration->GetVariable("$temp", 2);

			auto addressIn1 = new PTX::MemoryAddress<B, PTX::BitType<S>, PTX::ParameterSpace>(paramIn1);
			auto addressIn2 = new PTX::MemoryAddress<B, PTX::BitType<S>, PTX::ParameterSpace>(paramIn2);
			auto addressOut = new PTX::MemoryAddress<B, PTX::BitType<S>, PTX::ParameterSpace>(paramOut);

			auto converted1 = ConversionGenerator::ConvertSource<PTX::BitType<S>, PTX::FloatType<S>>(this->m_builder, src1);
			auto converted2 = ConversionGenerator::ConvertSource<PTX::BitType<S>, PTX::FloatType<S>>(this->m_builder, src2);

			this->m_builder.AddStatement(new PTX::StoreInstruction<B, PTX::BitType<S>, PTX::ParameterSpace>(addressIn1, converted1));
			this->m_builder.AddStatement(new PTX::StoreInstruction<B, PTX::BitType<S>, PTX::ParameterSpace>(addressIn2, converted2));
			this->m_builder.AddStatement(new PTX::CallInstruction<typename PTX::ExternalMathFunctions::BinaryFunction<S>::Signature>(function, paramOut, paramIn1, paramIn2));
			this->m_builder.AddStatement(new PTX::LoadInstruction<B, PTX::BitType<S>, PTX::ParameterSpace>(new PTX::BitRegisterAdapter<PTX::FloatType, S>(target), addressOut));
		}

		this->m_builder.CloseScope();
	}

private:
	PTX::ExternalMathFunctions::BinaryFunction<S> *GetFunction(ExternalBinaryOperation binaryOp) const
	{
		switch (binaryOp)
		{
			case ExternalBinaryOperation::Power:
				return PTX::ExternalMathFunctions::pow<S>;
			case ExternalBinaryOperation::Modulo:
				return PTX::ExternalMathFunctions::mod<S>;
			default:
				BuiltinGenerator<B, PTX::FloatType<S>>::Unimplemented("external function " + ExternalBinaryOperationString(binaryOp));
		}
	}

	ExternalBinaryOperation m_binaryOp;
};

}
