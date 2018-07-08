#pragma once

#include "Codegen/Generators/Expressions/Builtins/BuiltinGenerator.h"

#include "Codegen/Generators/TypeDispatch.h"
#include "Codegen/Generators/TypeUtils.h"

#include "PTX/Instructions/Comparison/SetPredicateInstruction.h"

namespace Codegen {

enum class ComparisonOperator {
	Equal,
	NotEqual,
	Less,
	LessEqual,
	Greater,
	GreaterEqual
};

template<PTX::Bits B, class T>
class ComparisonGenerator : public BuiltinGenerator<B, T>
{
public:
	ComparisonGenerator(const PTX::Register<T> *target, Builder *builder, ComparisonOperator comparisonOp) : BuiltinGenerator<B, T>(target, builder), m_comparisonOp(comparisonOp) {}

	void Generate(const HorseIR::CallExpression *call) override
	{
		std::cerr << "[ERROR] Unsupported type for builtin comparison function " + call->GetName() << std::endl;
		std::exit(EXIT_FAILURE);
	}

private:
	ComparisonOperator m_comparisonOp;
};

template<PTX::Bits B>
class ComparisonGenerator<B, PTX::PredicateType> : public BuiltinGenerator<B, PTX::PredicateType>
{
public:
	using NodeType = HorseIR::CallExpression;

	ComparisonGenerator(const PTX::Register<PTX::PredicateType> *target, Builder *builder, ComparisonOperator comparisonOp) : BuiltinGenerator<B, PTX::PredicateType>(target, builder), m_comparisonOp(comparisonOp) {}

	void Generate(const HorseIR::CallExpression *call) override
	{
		auto arg1 = call->GetArgument(0);
		auto arg2 = call->GetArgument(1);
		auto type = WidestType(arg1->GetType(), arg2->GetType());
		Dispatch<ComparisonGenerator<B, PTX::PredicateType>>(this, type, call);
	}

	using BuiltinGenerator<B, PTX::PredicateType>::Generate;

	template<class T>
	void Generate(const HorseIR::CallExpression *call)
	{
		if constexpr(PTX::is_comparable_type<T>::value)
		{
			OperandGenerator<B, T> opGen(this->m_builder);
			auto src1 = opGen.GenerateOperand(call->GetArgument(0));
			auto src2 = opGen.GenerateOperand(call->GetArgument(1));
			this->m_builder->AddStatement(new PTX::SetPredicateInstruction<T>(this->m_target, src1, src2, PTXOp<T>(m_comparisonOp)));
		}
		else
		{
			std::cerr << "[ERROR] Unsupported type for comparison operation" << std::endl;
			std::exit(EXIT_FAILURE);
		}
	}

private:
	template<class T>
	static typename T::ComparisonOperator PTXOp(ComparisonOperator comparisonOp)
	{
		switch (comparisonOp)
		{
			case ComparisonOperator::Greater:
				return T::ComparisonOperator::Greater;
			case ComparisonOperator::GreaterEqual:
				return T::ComparisonOperator::GreaterEqual;
			case ComparisonOperator::Less:
				return T::ComparisonOperator::Less;
			case ComparisonOperator::LessEqual:
				return T::ComparisonOperator::LessEqual;
			case ComparisonOperator::Equal:
				return T::ComparisonOperator::Equal;
			case ComparisonOperator::NotEqual:
				return T::ComparisonOperator::NotEqual;
			default:
				std::cerr << "[ERROR] Unsupported comparison operation" << std::endl;
				std::exit(EXIT_FAILURE);
		}

	}

	ComparisonOperator m_comparisonOp;
};

}
