#pragma once

#include "Codegen/Generators/Expressions/Builtins/BuiltinGenerator.h"

#include "Codegen/Generators/TypeDispatch.h"
#include "Codegen/Generators/TypeUtils.h"

namespace Codegen {

enum class TrigonometryOperation {
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

template<PTX::Bits B, class T>
class TrigonometryGenerator : public BuiltinGenerator<B, T>
{
public:
	TrigonometryGenerator(const PTX::Register<T> *target, Builder *builder, TrigonometryOperation trigOp) : BuiltinGenerator<B, T>(target, builder), m_trigOp(trigOp) {}

	void Generate(const HorseIR::CallExpression *call) override
	{
		//TODO: Work on error messages
		std::cerr << "[ERROR] Unsupported type for builtin trigonometric function " + call->GetName() << std::endl;
		std::exit(EXIT_FAILURE);
	}

private:
	TrigonometryOperation m_trigOp;
};

template<PTX::Bits B>
class TrigonometryGenerator<B, PTX::Float32Type> : public BuiltinGenerator<B, PTX::Float32Type>
{
public:
	TrigonometryGenerator(const PTX::Register<PTX::Float32Type> *target, Builder *builder, TrigonometryOperation trigOp) : BuiltinGenerator<B, PTX::Float32Type>(target, builder), m_trigOp(trigOp) {}

	void Generate(const HorseIR::CallExpression *call) override
	{
		OperandGenerator<B, PTX::Float32Type> opGen(this->m_builder);
		auto src = opGen.GenerateOperand(call->GetArgument(0));
		switch (m_trigOp)
		{
			//TODO: Add trigonometric functions, potentially linking external
			default:
				std::cerr << "[ERROR] Unsupported builtin trigonometric function " + call->GetName() << std::endl;
				std::exit(EXIT_FAILURE);
		}
	}

private:
	TrigonometryOperation m_trigOp;
};

}
