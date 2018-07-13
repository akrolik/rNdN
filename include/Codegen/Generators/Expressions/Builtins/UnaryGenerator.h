#pragma once

#include "Codegen/Generators/Expressions/Builtins/BuiltinGenerator.h"

#include "Codegen/Generators/Expressions/OperandGenerator.h"

#include "PTX/Instructions/Arithmetic/AbsoluteInstruction.h"
#include "PTX/Instructions/Arithmetic/NegateInstruction.h"
#include "PTX/Instructions/Logical/NotInstruction.h"
#include "PTX/Operands/Operand.h"
#include "PTX/Statements/BlockStatement.h"

namespace Codegen {

enum class UnaryOperation {
	// Arithmetic
	Absolute,
	Negate,

	// Logical
	Not
};

template<PTX::Bits B, class T>
class UnaryGenerator : public BuiltinGenerator<B, T>
{
public:
	UnaryGenerator(const PTX::Register<T> *target, Builder *builder, UnaryOperation unaryOp) : BuiltinGenerator<B, T>(target, builder), m_unaryOp(unaryOp) {}

	void Generate(const HorseIR::CallExpression *call) override
	{
		OperandGenerator<B, T> opGen(this->m_builder);
		auto src = opGen.GenerateOperand(call->GetArgument(0));

		switch (m_unaryOp)
		{
			case UnaryOperation::Absolute:
				GenerateInstruction<PTX::AbsoluteInstruction>(src);
				break;
			case UnaryOperation::Negate:
				GenerateInstruction<PTX::NegateInstruction>(src);
				break;
			case UnaryOperation::Not:
				GenerateInstruction<PTX::NotInstruction>(src);
				break;
			default:
				BuiltinGenerator<B, T>::Unimplemented(call);
		}
	}

	template<template<class, bool = true> class Op>
	void GenerateInstruction(const PTX::TypedOperand<T> *src)
	{
		if constexpr(Op<T, false>::TypeSupported)
		{
			this->m_builder->AddStatement(new Op<T>(this->m_target, src));
		}
		else
		{
			BuiltinGenerator<B, T>::Unimplemented(Op<T, false>::Mnemonic() + " instruction");
		}
	}

private:
	UnaryOperation m_unaryOp;
};

template<PTX::Bits B>
class UnaryGenerator<B, PTX::Int8Type> : public BuiltinGenerator<B, PTX::Int8Type>
{
public:
	UnaryGenerator(const PTX::Register<PTX::Int8Type> *target, Builder *builder, UnaryOperation unaryOp) : BuiltinGenerator<B, PTX::Int8Type>(target, builder), m_unaryOp(unaryOp) {}

	void Generate(const HorseIR::CallExpression *call) override
	{
		auto block = new PTX::BlockStatement();
		auto resources = this->m_builder->OpenScope(block);

		auto temp = resources->template AllocateRegister<PTX::Int16Type, ResourceKind::Internal>("temp");

		UnaryGenerator<B, PTX::Int16Type> gen(temp, this->m_builder, m_unaryOp);
		gen.Generate(call);

		this->m_builder->AddStatement(new PTX::ConvertInstruction<PTX::Int8Type, PTX::Int16Type>(this->m_target, temp));

		this->m_builder->CloseScope();
		this->m_builder->AddStatement(block);
	}

private:
	UnaryOperation m_unaryOp;
};

}
