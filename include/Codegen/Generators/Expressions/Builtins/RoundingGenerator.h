#pragma once

#include "Codegen/Generators/Expressions/Builtins/BuiltinGenerator.h"

#include "Codegen/Builder.h"
#include "Codegen/Generators/TypeDispatch.h"
#include "Codegen/Generators/Expressions/OperandCompressionGenerator.h"
#include "Codegen/Generators/Expressions/OperandGenerator.h"

#include "HorseIR/Tree/Tree.h"

#include "PTX/PTX.h"

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
	RoundingGenerator(Builder& builder, RoundingOperation roundOp) : BuiltinGenerator<B, T>(builder), m_roundOp(roundOp) {}

private:
	RoundingOperation m_roundOp;
};

template<PTX::Bits B, PTX::Bits S>
class RoundingGenerator<B, PTX::IntType<S>> : public BuiltinGenerator<B, PTX::IntType<S>>
{
public:
	using NodeType = HorseIR::CallExpression;

	RoundingGenerator(Builder& builder, RoundingOperation roundOp) : BuiltinGenerator<B, PTX::IntType<S>>(builder), m_roundOp(roundOp) {}

	const PTX::Register<PTX::PredicateType> *GenerateCompressionPredicate(const std::vector<HorseIR::Operand *>& arguments) override
	{
		return OperandCompressionGenerator::UnaryCompressionRegister(this->m_builder, arguments);
	}

	const PTX::Register<PTX::IntType<S>> *Generate(const HorseIR::LValue *target, const std::vector<HorseIR::Operand *>& arguments) override
	{
		DispatchType(*this, arguments.at(0)->GetType(), target, arguments);
		return m_targetRegister;
	}

	template<class T>
	void Generate(const HorseIR::LValue *target, const std::vector<HorseIR::Operand *>& arguments)
	{
		if constexpr(PTX::is_float_type<T>::value)
		{
			OperandGenerator<B, T> opGen(this->m_builder);
			auto src = opGen.GenerateOperand(arguments.at(0), OperandGenerator<B, T>::LoadKind::Vector);
			m_targetRegister = this->GenerateTargetRegister(target, arguments);

			auto conversion = new PTX::ConvertInstruction<PTX::IntType<S>, T>(m_targetRegister, src);
			conversion->SetRoundingMode(PTXOp<T>(m_roundOp));
			this->m_builder.AddStatement(conversion);
		}
		else
		{
			BuiltinGenerator<B, T>::Unimplemented("non-float rouding");
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

	const PTX::Register<PTX::IntType<S>> *m_targetRegister = nullptr;
};

}
