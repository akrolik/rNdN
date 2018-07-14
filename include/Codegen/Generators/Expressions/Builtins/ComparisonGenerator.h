#pragma once

#include "Codegen/Generators/Expressions/Builtins/BuiltinGenerator.h"

#include "Codegen/Generators/TypeDispatch.h"
#include "Codegen/Generators/TypeUtils.h"

#include "PTX/Instructions/Comparison/SelectInstruction.h"
#include "PTX/Instructions/Comparison/SetPredicateInstruction.h"
#include "PTX/Instructions/Data/MoveInstruction.h"

namespace Codegen {

enum class ComparisonOperator {
	Equal,
	NotEqual,
	Less,
	LessEqual,
	Greater,
	GreaterEqual,
	Sign
};

static std::string ComparisonOperatorString(ComparisonOperator comparisonOp)
{
	switch (comparisonOp)
	{
		case ComparisonOperator::Equal:
			return "=";
		case ComparisonOperator::NotEqual:
			return "!=";
		case ComparisonOperator::Less:
			return "<";
		case ComparisonOperator::LessEqual:
			return "<=";
		case ComparisonOperator::Greater:
			return ">";
		case ComparisonOperator::GreaterEqual:
			return ">=";
		case ComparisonOperator::Sign:
			return "signum";
	}
	return "<unknown>";
}

template<PTX::Bits B, class T, typename Enable = void>
class ComparisonGenerator : public BuiltinGenerator<B, T>
{
public:
	ComparisonGenerator(const PTX::Register<T> *target, Builder *builder, ComparisonOperator comparisonOp) : BuiltinGenerator<B, T>(target, builder), m_comparisonOp(comparisonOp) {}

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
		Dispatch(*this, type, call);
	}

	using BuiltinGenerator<B, PTX::PredicateType>::Generate;

	template<class T>
	void Generate(const HorseIR::CallExpression *call)
	{
		OperandGenerator<B, T> opGen(this->m_builder);
		auto src1 = opGen.GenerateOperand(call->GetArgument(0));
		auto src2 = opGen.GenerateOperand(call->GetArgument(1));
		Generate(this->m_target, src1, src2);
	}

	template<class T>
	void Generate(const PTX::Register<PTX::PredicateType> *target, const PTX::TypedOperand<T> *src1, const PTX::TypedOperand<T> *src2)
	{
		if constexpr(PTX::is_comparable_type<T>::value)
		{
			this->m_builder->AddStatement(new PTX::SetPredicateInstruction<T>(target, src1, src2, PTXOp<T>(m_comparisonOp)));
		}
		else
		{
			//TODO: Error
			BuiltinGenerator<B, T>::Unimplemented("string");
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
				//TODO: Error
				BuiltinGenerator<B, T>::Unimplemented("comparison operator " + ComparisonOperatorString(comparisonOp));
		}

	}

	ComparisonOperator m_comparisonOp;
};

template<PTX::Bits B, PTX::Bits S>
class ComparisonGenerator<B, PTX::IntType<S>, std::enable_if_t<S == PTX::Bits::Bits32 || S == PTX::Bits::Bits64>> : public BuiltinGenerator<B, PTX::IntType<S>>
{
public:
	using NodeType = HorseIR::CallExpression;

	ComparisonGenerator(const PTX::Register<PTX::IntType<S>> *target, Builder *builder, ComparisonOperator comparisonOp) : BuiltinGenerator<B, PTX::IntType<S>>(target, builder), m_comparisonOp(comparisonOp) {}

	void Generate(const HorseIR::CallExpression *call) override
	{
		if (m_comparisonOp == ComparisonOperator::Sign)
		{
			auto arg = call->GetArgument(0);
			Dispatch(*this, arg->GetType(), call);
		}
		else
		{
			//TODO: Error
			BuiltinGenerator<B, PTX::IntType<S>>::Unimplemented("comparison operator " + ComparisonOperatorString(m_comparisonOp));
		}
	}

	using BuiltinGenerator<B, PTX::IntType<S>>::Generate;

	template<class T>
	void Generate(const HorseIR::CallExpression *call)
	{
		auto block = new PTX::BlockStatement();
		this->m_builder->AddStatement(block);
		auto resources = this->m_builder->OpenScope(block);

		OperandGenerator<B, T> opGen(this->m_builder);
		auto src = opGen.GenerateOperand(call->GetArgument(0));

		auto tempP = resources->template AllocateRegister<PTX::PredicateType, ResourceKind::Internal>("p");
		auto tempQ = resources->template AllocateRegister<PTX::PredicateType, ResourceKind::Internal>("q");

		ComparisonGenerator<B, PTX::PredicateType> gen1(nullptr, this->m_builder, ComparisonOperator::Equal);
		gen1.template Generate<T>(tempP, src, new PTX::Value<T>(0));

		auto move = new PTX::MoveInstruction<PTX::IntType<S>>(this->m_target, new PTX::Value<PTX::IntType<S>>(0));
		move->SetPredicate(tempP);
		this->m_builder->AddStatement(move);

		ComparisonGenerator<B, PTX::PredicateType> gen2(nullptr, this->m_builder, ComparisonOperator::Greater);
		gen2.template Generate<T>(tempQ, src, new PTX::Value<T>(0));

		auto select = new PTX::SelectInstruction<PTX::IntType<S>>(this->m_target, new PTX::Value<PTX::IntType<S>>(1), new PTX::Value<PTX::IntType<S>>(-1), tempQ);
		select->SetPredicate(tempP, true);
		this->m_builder->AddStatement(select);

		this->m_builder->CloseScope();
	}

private:
	ComparisonOperator m_comparisonOp;
};

}
