#pragma once

#include "Codegen/Generators/Expressions/Builtins/BuiltinGenerator.h"

#include "Codegen/Generators/Expressions/OperandGenerator.h"

#include "PTX/Instructions/Arithmetic/AbsoluteInstruction.h"
#include "PTX/Instructions/Arithmetic/NegateInstruction.h"
#include "PTX/Instructions/Arithmetic/ReciprocalInstruction.h"
#include "PTX/Instructions/Logical/NotInstruction.h"
#include "PTX/Operands/Operand.h"
#include "PTX/Statements/BlockStatement.h"

namespace Codegen {

enum class UnaryOperation {
	// Arithmetic
	Absolute,
	Negate,
	Reciprocal,

	// Logical
	Not,

	// Numeric
	Pi
};

static std::string UnaryOperationString(UnaryOperation unaryOp)
{
	switch (unaryOp)
	{
		case UnaryOperation::Absolute:
			return "abs";
		case UnaryOperation::Negate:
			return "neg";
		case UnaryOperation::Reciprocal:
			return "recip";
		case UnaryOperation::Not:
			return "not";
		case UnaryOperation::Pi:
			return "pi";
	}
	return "<unknown>";
}

template<PTX::Bits B, class T>
class UnaryGenerator : public BuiltinGenerator<B, T>
{
public:
	UnaryGenerator(Builder *builder, UnaryOperation unaryOp) : BuiltinGenerator<B, T>(builder), m_unaryOp(unaryOp) {}

	void Generate(const PTX::Register<T> *target, const HorseIR::CallExpression *call) override
	{
		OperandGenerator<B, T> opGen(this->m_builder);
		auto src = opGen.GenerateOperand(call->GetArgument(0));
		Generate(target, src);
	}

	void Generate(const PTX::Register<T> *target, const PTX::TypedOperand<T> *src)
	{
		switch (m_unaryOp)
		{
			case UnaryOperation::Absolute:
				GenerateInstruction<PTX::AbsoluteInstruction>(target, src);
				break;
			case UnaryOperation::Negate:
				GenerateInstruction<PTX::NegateInstruction>(target, src);
				break;
			case UnaryOperation::Reciprocal:
				GenerateInstruction<PTX::ReciprocalInstruction>(target, src);
				break;
			case UnaryOperation::Not:
				GenerateInstruction<PTX::NotInstruction>(target, src);
				break;
			case UnaryOperation::Pi:
				GeneratePi(target, src);
				break;
			default:
				BuiltinGenerator<B, T>::Unimplemented("unary operation " + UnaryOperationString(m_unaryOp));
		}
	}

	template<template<class, bool = true> class Op>
	void GenerateInstruction(const PTX::Register<T> *target, const PTX::TypedOperand<T> *src)
	{
		if constexpr(Op<T, false>::TypeSupported)
		{
			this->m_builder->AddStatement(new Op<T>(target, src));
		}
		else
		{
			BuiltinGenerator<B, T>::Unimplemented(Op<T, false>::Mnemonic() + " instruction");
		}
	}

private:
	void GeneratePi(const PTX::Register<T> *target, const PTX::TypedOperand<T> *src);

	UnaryOperation m_unaryOp;
};

template<PTX::Bits B>
class UnaryGenerator<B, PTX::Int8Type> : public BuiltinGenerator<B, PTX::Int8Type>
{
public:
	UnaryGenerator(Builder *builder, UnaryOperation unaryOp) : BuiltinGenerator<B, PTX::Int8Type>(builder), m_unaryOp(unaryOp) {}

	void Generate(const PTX::Register<PTX::Int8Type> *target, const HorseIR::CallExpression *call) override
	{
		auto block = new PTX::BlockStatement();
		auto resources = this->m_builder->OpenScope(block);

		auto temp = resources->template AllocateTemporary<PTX::Int16Type>();

		UnaryGenerator<B, PTX::Int16Type> gen(this->m_builder, m_unaryOp);
		gen.Generate(temp, call);

		this->m_builder->AddStatement(new PTX::ConvertInstruction<PTX::Int8Type, PTX::Int16Type>(target, temp));

		this->m_builder->CloseScope();
		this->m_builder->AddStatement(block);
	}

private:
	UnaryOperation m_unaryOp;
};

}

#include "Codegen/Generators/Expressions/Builtins/BinaryGenerator.h"

namespace Codegen {

template<PTX::Bits B, class T>
void UnaryGenerator<B, T>::GeneratePi(const PTX::Register<T> *target, const PTX::TypedOperand<T> *src)
{
	if constexpr(std::is_same<T, PTX::Float32Type>::value || std::is_same<T, PTX::Float64Type>::value)
	{
		#define CUDART_PI_F 3.141592654f

		BinaryGenerator<B, T> gen(this->m_builder, BinaryOperation::Multiply);
		gen.Generate(target, src, new PTX::Value<T>(CUDART_PI_F));
	}
	else
	{
		BuiltinGenerator<B, T>::Unimplemented("unary operation " + UnaryOperationString(m_unaryOp));
	}
}

}
