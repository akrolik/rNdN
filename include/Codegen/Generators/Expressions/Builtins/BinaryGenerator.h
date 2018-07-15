#pragma once

#include "Codegen/Generators/Expressions/Builtins/BuiltinGenerator.h"

#include "Codegen/Generators/Expressions/OperandGenerator.h"

#include "PTX/Instructions/Arithmetic/AddInstruction.h"
#include "PTX/Instructions/Arithmetic/DivideInstruction.h"
#include "PTX/Instructions/Arithmetic/MultiplyInstruction.h"
#include "PTX/Instructions/Arithmetic/SubtractInstruction.h"
#include "PTX/Instructions/Logical/AndInstruction.h"
#include "PTX/Instructions/Logical/OrInstruction.h"
#include "PTX/Instructions/Logical/XorInstruction.h"
#include "PTX/Operands/Operand.h"
#include "PTX/Statements/BlockStatement.h"

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
	BinaryGenerator(Builder *builder, BinaryOperation binaryOp) : BuiltinGenerator<B, T>(builder), m_binaryOp(binaryOp) {}

	void Generate(const PTX::Register<T> *target, const HorseIR::CallExpression *call) override
	{
		OperandGenerator<B, T> opGen(this->m_builder);
		auto src1 = opGen.GenerateOperand(call->GetArgument(0));
		auto src2 = opGen.GenerateOperand(call->GetArgument(1));
		Generate(target, src1, src2);
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
				//TODO: Implement @nand
				break;
			case BinaryOperation::Nor:
				//TODO: Implement @nor
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
				this->m_builder->AddStatement(new Op<T>(target, src1, src2, PTX::Float64Type::RoundingMode::Nearest));
			}
			else
			{
				this->m_builder->AddStatement(new Op<T>(target, src1, src2));
			}
		}
		else
		{
			BuiltinGenerator<B, T>::Unimplemented(Op<T, false>::Mnemonic() + " instruction");
		}
	}

private:
	BinaryOperation m_binaryOp;
};

template<PTX::Bits B>
class BinaryGenerator<B, PTX::Int8Type> : public BuiltinGenerator<B, PTX::Int8Type>
{
public:
	BinaryGenerator(Builder *builder, BinaryOperation binaryOp) : BuiltinGenerator<B, PTX::Int8Type>(builder), m_binaryOp(binaryOp) {}

	void Generate(const PTX::Register<PTX::Int8Type> *target, const HorseIR::CallExpression *call) override
	{
		auto block = new PTX::BlockStatement();
		auto resources = this->m_builder->OpenScope(block);
		this->m_builder->AddStatement(block);

		auto temp = resources->template AllocateRegister<PTX::Int16Type, ResourceKind::Internal>("temp");

		BinaryGenerator<B, PTX::Int16Type> gen(this->m_builder, m_binaryOp);
		gen.Generate(temp, call);

		this->m_builder->AddStatement(new PTX::ConvertInstruction<PTX::Int8Type, PTX::Int16Type>(target, temp));

		this->m_builder->CloseScope();
	}

private:
	BinaryOperation m_binaryOp;
};

}
