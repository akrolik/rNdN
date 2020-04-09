#pragma once

#include "Codegen/Generators/Expressions/Builtins/BuiltinGenerator.h"

#include "Codegen/Builder.h"
#include "Codegen/Generators/TypeDispatch.h"
#include "Codegen/Generators/Expressions/OperandCompressionGenerator.h"
#include "Codegen/Generators/Expressions/OperandGenerator.h"

#include "HorseIR/Tree/Tree.h"
#include "HorseIR/Utils/TypeUtils.h"

#include "PTX/PTX.h"

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
	ComparisonGenerator(Builder& builder, ComparisonOperator comparisonOp) : BuiltinGenerator<B, T>(builder), m_comparisonOp(comparisonOp) {}

	std::string Name() const override { return "ComparisonGenerator"; }

private:
	ComparisonOperator m_comparisonOp;
};

template<PTX::Bits B>
class ComparisonGenerator<B, PTX::PredicateType> : public BuiltinGenerator<B, PTX::PredicateType>
{
public:
	ComparisonGenerator(Builder& builder, ComparisonOperator comparisonOp) : BuiltinGenerator<B, PTX::PredicateType>(builder), m_comparisonOp(comparisonOp) {}

	std::string Name() const override { return "ComparisonGenerator"; }

	const PTX::Register<PTX::PredicateType> *GenerateCompressionPredicate(const std::vector<HorseIR::Operand *>& arguments) override
	{
		return OperandCompressionGenerator::BinaryCompressionRegister(this->m_builder, arguments);
	}

	const PTX::Register<PTX::PredicateType> *Generate(const HorseIR::LValue *target, const std::vector<HorseIR::Operand *>& arguments) override
	{
		auto type = HorseIR::TypeUtils::WidestType(arguments.at(0)->GetType(), arguments.at(1)->GetType());
		DispatchType(*this, type, target, arguments);
		return m_targetRegister;
	}

	template<class T>
	void GenerateVector(const HorseIR::LValue *target, const std::vector<HorseIR::Operand *>& arguments)
	{
		//TODO: Predicate type should sometimes compare using bit operations

		if constexpr(std::is_same<T, PTX::PredicateType>::value || std::is_same<T, PTX::Int8Type>::value)
		{
			// Conversions require at least 16 bits

			GenerateVector<PTX::Int16Type>(target, arguments);
		}
		else
		{
			OperandGenerator<B, T> opGen(this->m_builder);
			auto src1 = opGen.GenerateOperand(arguments.at(0), OperandGenerator<B, T>::LoadKind::Vector);
			auto src2 = opGen.GenerateOperand(arguments.at(1), OperandGenerator<B, T>::LoadKind::Vector);
			m_targetRegister = this->GenerateTargetRegister(target, arguments);
			Generate(m_targetRegister, src1, src2);
		}
	}

	template<class T>
	void GenerateList(const HorseIR::LValue *target, const std::vector<HorseIR::Operand *>& arguments)
	{
		if (this->m_builder.GetInputOptions().IsVectorGeometry())
		{
			BuiltinGenerator<B, PTX::PredicateType>::Unimplemented("list-in-vector");
		}

		// Lists are handled by the vector code through a projection

		GenerateVector<T>(target, arguments);
	}

	template<class T>
	void GenerateTuple(unsigned int index, const HorseIR::LValue *target, const std::vector<HorseIR::Operand *>& arguments)
	{
		BuiltinGenerator<B, PTX::PredicateType>::Unimplemented("list-in-vector");
	}

	template<class T>
	void Generate(const PTX::Register<PTX::PredicateType> *target, const PTX::TypedOperand<T> *src1, const PTX::TypedOperand<T> *src2)
	{
		if constexpr(PTX::is_comparable_type<T>::value)
		{
			this->m_builder.AddStatement(new PTX::SetPredicateInstruction<T>(target, src1, src2, PTXOp<T>(m_comparisonOp)));
		}
		else
		{
			BuiltinGenerator<B, PTX::PredicateType>::Unimplemented("comparison operator " + ComparisonOperatorString(m_comparisonOp));
		}
	}

private:
	template<class T>
	typename T::ComparisonOperator PTXOp(ComparisonOperator comparisonOp) const
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
				BuiltinGenerator<B, PTX::PredicateType>::Unimplemented("comparison operator " + ComparisonOperatorString(comparisonOp));
		}

	}

	ComparisonOperator m_comparisonOp;
	const PTX::Register<PTX::PredicateType> *m_targetRegister = nullptr;
};

template<PTX::Bits B, PTX::Bits S>
class ComparisonGenerator<B, PTX::IntType<S>, std::enable_if_t<S == PTX::Bits::Bits32 || S == PTX::Bits::Bits64>> : public BuiltinGenerator<B, PTX::IntType<S>>
{
public:
	ComparisonGenerator(Builder& builder, ComparisonOperator comparisonOp) : BuiltinGenerator<B, PTX::IntType<S>>(builder), m_comparisonOp(comparisonOp) {}

	std::string Name() const override { return "ComparisonGenerator"; }

	const PTX::Register<PTX::PredicateType> *GenerateCompressionPredicate(const std::vector<HorseIR::Operand *>& arguments) override
	{
		return OperandCompressionGenerator::BinaryCompressionRegister(this->m_builder, arguments);
	}

	const PTX::Register<PTX::IntType<S>> *Generate(const HorseIR::LValue *target, const std::vector<HorseIR::Operand *>& arguments) override
	{
		if (m_comparisonOp == ComparisonOperator::Sign)
		{
			DispatchType(*this, arguments.at(0)->GetType(), target, arguments);
			return m_targetRegister;
		}
		else
		{
			BuiltinGenerator<B, PTX::IntType<S>>::Unimplemented("comparison operator " + ComparisonOperatorString(m_comparisonOp));
		}
	}

	template<class T>
	void GenerateVector(const HorseIR::LValue *target, const std::vector<HorseIR::Operand *>& arguments)
	{
		auto block = new PTX::BlockStatement();
		this->m_builder.AddStatement(block);
		auto resources = this->m_builder.OpenScope(block);

		OperandGenerator<B, T> opGen(this->m_builder);
		auto src = opGen.GenerateOperand(arguments.at(0), OperandGenerator<B, T>::LoadKind::Vector);
		m_targetRegister = this->GenerateTargetRegister(target, arguments);

		auto tempP = resources->template AllocateTemporary<PTX::PredicateType>();
		auto tempQ = resources->template AllocateTemporary<PTX::PredicateType>();

		ComparisonGenerator<B, PTX::PredicateType> gen1(this->m_builder, ComparisonOperator::Equal);
		gen1.template Generate<T>(tempP, src, new PTX::Value<T>(0));

		auto move = new PTX::MoveInstruction<PTX::IntType<S>>(m_targetRegister, new PTX::Value<PTX::IntType<S>>(0));
		move->SetPredicate(tempP);
		this->m_builder.AddStatement(move);

		ComparisonGenerator<B, PTX::PredicateType> gen2(this->m_builder, ComparisonOperator::Greater);
		gen2.template Generate<T>(tempQ, src, new PTX::Value<T>(0));

		auto select = new PTX::SelectInstruction<PTX::IntType<S>>(m_targetRegister, new PTX::Value<PTX::IntType<S>>(1), new PTX::Value<PTX::IntType<S>>(-1), tempQ);
		select->SetPredicate(tempP, true);
		this->m_builder.AddStatement(select);

		this->m_builder.CloseScope();
	}

	template<class T>
	void GenerateList(const HorseIR::LValue *target, const std::vector<HorseIR::Operand *>& arguments)
	{
		if (this->m_builder.GetInputOptions().IsVectorGeometry())
		{
			BuiltinGenerator<B, PTX::IntType<S>>::Unimplemented("list-in-vector");
		}

		// Lists are handled by the vector code through a projection

		GenerateVector<T>(target, arguments);
	}

	template<class T>
	void GenerateTuple(unsigned int index, const HorseIR::LValue *target, const std::vector<HorseIR::Operand *>& arguments)
	{
		BuiltinGenerator<B, PTX::IntType<S>>::Unimplemented("list-in-vector");
	}

private:
	ComparisonOperator m_comparisonOp;
	
	const PTX::Register<PTX::IntType<S>> *m_targetRegister = nullptr;
};

}
