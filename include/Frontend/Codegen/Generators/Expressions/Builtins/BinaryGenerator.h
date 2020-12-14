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

enum class BinaryOperation {
	// Arithmetic
	Plus,
	Minus,
	Multiply,
	Divide,

	// Logical
	And,
	Or,
	Nand,
	Nor,
	Xor
};

static std::string BinaryOperationString(BinaryOperation binaryOp)
{
	switch (binaryOp)
	{
		case BinaryOperation::Plus:
			return "plus";
		case BinaryOperation::Minus:
			return "minus";
		case BinaryOperation::Multiply:
			return "mul";
		case BinaryOperation::Divide:
			return "div";
		case BinaryOperation::And:
			return "and";
		case BinaryOperation::Or:
			return "or";
		case BinaryOperation::Nand:
			return "nand";
		case BinaryOperation::Nor:
			return "nor";
		case BinaryOperation::Xor:
			return "xor";
	}
	return "<unknown>";
}

template<PTX::Bits B, class T>
class BinaryGenerator : public BuiltinGenerator<B, T>
{
public:
	BinaryGenerator(Builder& builder, BinaryOperation binaryOp) : BuiltinGenerator<B, T>(builder), m_binaryOp(binaryOp) {}

	std::string Name() const override { return "BinaryGenerator"; }

	PTX::Register<PTX::PredicateType> *GenerateCompressionPredicate(const std::vector<const HorseIR::Operand *>& arguments) override
	{
		return OperandCompressionGenerator::BinaryCompressionRegister(this->m_builder, arguments);
	}

	PTX::Register<T> *Generate(const HorseIR::LValue *target, const std::vector<const HorseIR::Operand *>& arguments) override
	{
		auto targetRegister = this->GenerateTargetRegister(target, arguments);
		if constexpr(std::is_same<T, PTX::Int8Type>::value)
		{
			auto resources = this->m_builder.GetLocalResources();
			auto targetRegister16 = resources->template AllocateTemporary<PTX::Int16Type>();

			OperandGenerator<B, PTX::Int16Type> opGen(this->m_builder);
			auto src1 = opGen.GenerateOperand(arguments.at(0), OperandGenerator<B, PTX::Int16Type>::LoadKind::Vector);
			auto src2 = opGen.GenerateOperand(arguments.at(1), OperandGenerator<B, PTX::Int16Type>::LoadKind::Vector);

			BinaryGenerator<B, PTX::Int16Type> generator(this->m_builder, m_binaryOp);
			generator.Generate(targetRegister16, src1, src2);
			ConversionGenerator::ConvertSource<PTX::Int8Type, PTX::Int16Type>(this->m_builder, targetRegister, targetRegister16);
		}
		else
		{
			OperandGenerator<B, T> opGen(this->m_builder);
			auto src1 = opGen.GenerateOperand(arguments.at(0), OperandGenerator<B, T>::LoadKind::Vector);
			auto src2 = opGen.GenerateOperand(arguments.at(1), OperandGenerator<B, T>::LoadKind::Vector);
			Generate(targetRegister, src1, src2);
		}
		return targetRegister;
	}

	void Generate(PTX::Register<T> *target, PTX::TypedOperand<T> *src1, PTX::TypedOperand<T> *src2)
	{	
		switch (m_binaryOp)
		{
			case BinaryOperation::Plus:
				GenerateInstruction<PTX::AddInstruction>(target, src1, src2);
				break;
			case BinaryOperation::Minus:
				GenerateInstruction<PTX::SubtractInstruction>(target ,src1, src2);
				break;
			case BinaryOperation::Multiply:
				GenerateInstruction<PTX::MultiplyInstruction>(target, src1, src2);
				break;
			case BinaryOperation::Divide:
				GenerateInstruction<PTX::DivideInstruction>(target, src1, src2);
				break;
			case BinaryOperation::And:
				GenerateInstruction<PTX::AndInstruction>(target, src1, src2);
				break;
			case BinaryOperation::Or:
				GenerateInstruction<PTX::OrInstruction>(target, src1, src2);
				break;
			case BinaryOperation::Nand:
				GenerateInverseInstruction<PTX::AndInstruction>(target, src1, src2);
				break;
			case BinaryOperation::Nor:
				GenerateInverseInstruction<PTX::OrInstruction>(target, src1, src2);
				break;
			case BinaryOperation::Xor:
				GenerateInstruction<PTX::XorInstruction>(target, src1, src2);
				break;
			default:
				BuiltinGenerator<B, T>::Unimplemented("binary operation " + BinaryOperationString(m_binaryOp));
		}
	}

	template<template<class, bool = true> class Op>
	void GenerateInstruction(PTX::Register<T> *target, PTX::TypedOperand<T> *src1, PTX::TypedOperand<T> *src2)
	{
		if constexpr(Op<T, false>::TypeSupported)
		{
			if constexpr(std::is_same<Op<T>, PTX::DivideInstruction<PTX::Float64Type>>::value)
			{
				this->m_builder.AddStatement(new Op<T>(target, src1, src2, PTX::Float64Type::RoundingMode::Nearest));
			}
			else if constexpr(std::is_same<Op<T>, PTX::MultiplyInstruction<T>>::value && PTX::HalfModifier<T>::Enabled)
			{
				this->m_builder.AddStatement(new PTX::MultiplyInstruction<T>(target, src1, src2, PTX::HalfModifier<T>::Half::Lower));
			}
			else
			{
				this->m_builder.AddStatement(new Op<T>(target, src1, src2));
			}
		}
		else
		{
			BuiltinGenerator<B, T>::Unimplemented(Op<T, false>::Mnemonic() + " instruction");
		}
	}

private:
	template<template<class, bool = true> class Op>
	void GenerateInverseInstruction(PTX::Register<T> *target, PTX::TypedOperand<T> *src1, PTX::TypedOperand<T> *src2);

	BinaryOperation m_binaryOp;
};

}
}

#include "Frontend/Codegen/Generators/Expressions/Builtins/UnaryGenerator.h"

namespace Frontend {
namespace Codegen {

template<PTX::Bits B, class T>
template<template<class, bool = true> class Op>
void BinaryGenerator<B, T>::GenerateInverseInstruction(PTX::Register<T> *target, PTX::TypedOperand<T> *src1, PTX::TypedOperand<T> *src2)
{
	auto resources = this->m_builder.GetLocalResources();
	auto temp = resources->template AllocateTemporary<T>();

	GenerateInstruction<Op>(temp, src1, src2);

	UnaryGenerator<B, T> gen(this->m_builder, UnaryOperation::Not);
	gen.Generate(target, temp);
};

}
}
