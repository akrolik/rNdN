#pragma once

#include "Codegen/Generators/Expressions/Builtins/BuiltinGenerator.h"

#include "Codegen/Generators/TypeDispatch.h"
#include "Codegen/Generators/TypeUtils.h"

#include "PTX/Instructions/Data/ConvertInstruction.h"

namespace Codegen {

enum class RoundingOperation {
	Ceiling,
	Floor,
	Nearest
};

static std::string RoundingOperationString(RoundingOperation roundOp)
{
	switch (roundOp)
	{
		case RoundingOperation::Ceiling:
			return "ceil";
		case RoundingOperation::Floor:
			return "floor";
		case RoundingOperation::Nearest:
			return "round";
	}
	return "<unknown>";
}

template<PTX::Bits B, class T>
class RoundingGenerator : public BuiltinGenerator<B, T>
{
public:
	RoundingGenerator(Builder *builder, RoundingOperation roundOp) : BuiltinGenerator<B, T>(builder), m_roundOp(roundOp) {}

private:
	RoundingOperation m_roundOp;
};

template<PTX::Bits B, PTX::Bits S>
class RoundingGenerator<B, PTX::IntType<S>> : public BuiltinGenerator<B, PTX::IntType<S>>
{
public:
	using NodeType = HorseIR::CallExpression;

	RoundingGenerator(Builder *builder, RoundingOperation roundOp) : BuiltinGenerator<B, PTX::IntType<S>>(builder), m_roundOp(roundOp) {}

	void Generate(const PTX::Register<PTX::IntType<S>> *target, const HorseIR::CallExpression *call) override
	{
		auto arg = call->GetArgument(0);
		Codegen::DispatchType(*this, arg->GetType(), target, call);
	}

	template<class T>
	void Generate(const PTX::Register<PTX::IntType<S>> *target, const HorseIR::CallExpression *call)
	{
		if constexpr(PTX::is_float_type<T>::value)
		{
			OperandGenerator<B, T> opGen(this->m_builder);
			auto src = opGen.GenerateOperand(call->GetArgument(0));
			auto conversion = new PTX::ConvertInstruction<PTX::IntType<S>, T>(target, src);
			conversion->SetRoundingMode(PTXOp<T>(m_roundOp));
			this->m_builder->AddStatement(conversion);
		}
		else
		{
			BuiltinGenerator<B, T>::Unimplemented(call);
		}
	}

private:
	template<class T>
	static typename PTX::ConvertRoundingModifier<PTX::IntType<S>, T>::RoundingMode PTXOp(RoundingOperation roundOp)
	{
		switch (roundOp)
		{
			case RoundingOperation::Ceiling:
				return PTX::ConvertRoundingModifier<PTX::IntType<S>, T>::RoundingMode::PositiveInfinity;
			case RoundingOperation::Floor:
				return PTX::ConvertRoundingModifier<PTX::IntType<S>, T>::RoundingMode::NegativeInfinity;
			case RoundingOperation::Nearest:
				return PTX::ConvertRoundingModifier<PTX::IntType<S>, T>::RoundingMode::Nearest;
			default:
				BuiltinGenerator<B, T>::Unimplemented("rounding operator " + RoundingOperationString(roundOp));
		}

	}

	RoundingOperation m_roundOp;
};

}
