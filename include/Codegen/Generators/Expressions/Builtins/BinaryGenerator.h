#pragma once

#include "Codegen/Generators/Expressions/Builtins/BuiltinGenerator.h"

#include "Codegen/Builder.h"
#include "Codegen/Generators/Expressions/OperandCompressionGenerator.h"
#include "Codegen/Generators/Expressions/OperandGenerator.h"

#include "HorseIR/Tree/Tree.h"

#include "PTX/PTX.h"

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

	const PTX::Register<PTX::PredicateType> *GenerateCompressionPredicate(const std::vector<HorseIR::Operand *>& arguments) override
	{
		return OperandCompressionGenerator::BinaryCompressionRegister(this->m_builder, arguments);
	}

	const PTX::Register<T> *Generate(const HorseIR::LValue *target, const std::vector<HorseIR::Operand *>& arguments) override
	{
		OperandGenerator<B, T> opGen(this->m_builder);
		auto src1 = opGen.GenerateOperand(arguments.at(0), OperandGenerator<B, T>::LoadKind::Vector);
		auto src2 = opGen.GenerateOperand(arguments.at(1), OperandGenerator<B, T>::LoadKind::Vector);

		auto targetRegister = this->GenerateTargetRegister(target, arguments);
		Generate(targetRegister, src1, src2);

		return targetRegister;
	}

	void Generate(const PTX::Register<T> *target, const PTX::TypedOperand<T> *src1, const PTX::TypedOperand<T> *src2)
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
	void GenerateInstruction(const PTX::Register<T> *target, const PTX::TypedOperand<T> *src1, const PTX::TypedOperand<T> *src2)
	{
		if constexpr(Op<T, false>::TypeSupported)
		{
			if constexpr(std::is_same<Op<T>, PTX::DivideInstruction<PTX::Float64Type>>::value)
			{
				this->m_builder.AddStatement(new Op<T>(target, src1, src2, PTX::Float64Type::RoundingMode::Nearest));
			}
			else if constexpr(std::is_same<Op<T>, PTX::MultiplyInstruction<T>>::value && PTX::is_int_type<T>::value)
			{
				auto instruction = new Op<T>(target, src1, src2);
				instruction->SetLower(true);
				this->m_builder.AddStatement(instruction);
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
	void GenerateInverseInstruction(const PTX::Register<T> *target, const PTX::TypedOperand<T> *src1, const PTX::TypedOperand<T> *src2);

	BinaryOperation m_binaryOp;
};

template<PTX::Bits B>
class BinaryGenerator<B, PTX::Int8Type> : public BuiltinGenerator<B, PTX::Int8Type>
{
public:
	BinaryGenerator(Builder& builder, BinaryOperation binaryOp) : BuiltinGenerator<B, PTX::Int8Type>(builder), m_binaryOp(binaryOp) {}

	const PTX::Register<PTX::Int8Type> *Generate(const HorseIR::LValue *target, const std::vector<HorseIR::Operand *>& arguments) override
	{
		BinaryGenerator<B, PTX::Int16Type> gen(this->m_builder, m_binaryOp);
		auto temp = gen.Generate(target, arguments);

		auto targetRegister = this->GenerateTargetRegister(target, arguments);
		this->m_builder.AddStatement(new PTX::ConvertInstruction<PTX::Int8Type, PTX::Int16Type>(targetRegister, temp));

		return targetRegister;
	}

private:
	BinaryOperation m_binaryOp;
};

}

#include "Codegen/Generators/Expressions/Builtins/UnaryGenerator.h"

namespace Codegen {

template<PTX::Bits B, class T>
template<template<class, bool = true> class Op>
void BinaryGenerator<B, T>::GenerateInverseInstruction(const PTX::Register<T> *target, const PTX::TypedOperand<T> *src1, const PTX::TypedOperand<T> *src2)
{
	auto resources = this->m_builder.GetLocalResources();

	auto temp = resources->template AllocateTemporary<T>();
	GenerateInstruction<Op>(temp, src1, src2);

	UnaryGenerator<B, T> gen(this->m_builder, UnaryOperation::Not);
	gen.Generate(target, temp);
};

}
