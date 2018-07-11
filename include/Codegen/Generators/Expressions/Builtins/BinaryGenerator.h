#pragma once

#include "Codegen/Generators/Expressions/Builtins/BuiltinGenerator.h"

#include "Codegen/Generators/Expressions/OperandGenerator.h"

#include "PTX/Instructions/Arithmetic/AddInstruction.h"
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

template<PTX::Bits B, class T>
class BinaryGenerator : public BuiltinGenerator<B, T>
{
public:
	BinaryGenerator(const PTX::Register<T> *target, Builder *builder, BinaryOperation binaryOp) : BuiltinGenerator<B, T>(target, builder), m_binaryOp(binaryOp) {}

	void Generate(const HorseIR::CallExpression *call) override
	{
		OperandGenerator<B, T> opGen(this->m_builder);
		auto src1 = opGen.GenerateOperand(call->GetArgument(0));
		auto src2 = opGen.GenerateOperand(call->GetArgument(1));

		switch (m_binaryOp)
		{
			//TODO: Complete binary operations
			case BinaryOperation::Plus:
				GenerateInstruction<PTX::AddInstruction>(src1, src2);
				break;
			case BinaryOperation::Minus:
				GenerateInstruction<PTX::SubtractInstruction>(src1, src2);
				break;
			case BinaryOperation::Multiply:
				GenerateInstruction<PTX::MultiplyInstruction>(src1, src2);
				break;
			case BinaryOperation::And:
				GenerateInstruction<PTX::AndInstruction>(src1, src2);
				break;
			case BinaryOperation::Or:
				GenerateInstruction<PTX::OrInstruction>(src1, src2);
				break;
			case BinaryOperation::Xor:
				GenerateInstruction<PTX::XorInstruction>(src1, src2);
				break;
			default:
				BuiltinGenerator<B, T>::Unimplemented(call);
		}
	}

	template<template<class, bool = true> class Op>
	void GenerateInstruction(const PTX::TypedOperand<T> *src1, const PTX::TypedOperand<T> *src2)
	{
		if constexpr(Op<T, false>::TypeSupported)
		{
			this->m_builder->AddStatement(new Op<T>(this->m_target, src1, src2));
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
	BinaryGenerator(const PTX::Register<PTX::Int8Type> *target, Builder *builder, BinaryOperation binaryOp) : BuiltinGenerator<B, PTX::Int8Type>(target, builder), m_binaryOp(binaryOp) {}

	void Generate(const HorseIR::CallExpression *call) override
	{
		auto block = new PTX::BlockStatement();
		auto resources = this->m_builder->OpenScope(block);

		auto temp = resources->template AllocateRegister<PTX::Int16Type, ResourceKind::Internal>("temp");

		BinaryGenerator<B, PTX::Int16Type> gen(temp, this->m_builder, m_binaryOp);
		gen.Generate(call);

		this->m_builder->AddStatement(new PTX::ConvertInstruction<PTX::Int8Type, PTX::Int16Type>(this->m_target, temp));

		this->m_builder->CloseScope();
		this->m_builder->AddStatement(block);
	}

private:
	BinaryOperation m_binaryOp;
};

}
