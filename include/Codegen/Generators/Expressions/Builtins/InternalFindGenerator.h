#pragma once

#include "HorseIR/Traversal/ConstVisitor.h"
#include "Codegen/Generators/Expressions/Builtins/BuiltinGenerator.h"

#include "Codegen/Builder.h"
#include "Codegen/Generators/Expressions/OperandCompressionGenerator.h"
#include "Codegen/Generators/Expressions/OperandGenerator.h"
#include "Codegen/Generators/Indexing/DataSizeGenerator.h"

#include "HorseIR/Tree/Tree.h"

#include "PTX/PTX.h"

namespace Codegen {

template<PTX::Bits B, class T, class D>
class InternalFindGenerator : public BuiltinGenerator<B, D>, public HorseIR::ConstVisitor
{
public:
	using BuiltinGenerator<B, D>::BuiltinGenerator;

	std::string Name() const override { return "InternalFindGenerator"; }

	const PTX::Register<PTX::PredicateType> *GenerateCompressionPredicate(const std::vector<HorseIR::Operand *>& arguments) override
	{
		return OperandCompressionGenerator::UnaryCompressionRegister(this->m_builder, arguments);
	}

	const PTX::Register<D> *Generate(const HorseIR::LValue *target, const std::vector<HorseIR::Operand *>& arguments)
	{
		m_targetRegister = this->GenerateTargetRegister(target, arguments);

		// Initialize target register (@member->false)

		if constexpr(std::is_same<D, PTX::PredicateType>::value)
		{
			this->m_builder.AddStatement(new PTX::MoveInstruction<PTX::PredicateType>(m_targetRegister, new PTX::BoolValue(false)));
		}

		// Operating range in argument 0, possibilities in argument 1

		OperandGenerator<B, T> opGen(this->m_builder);
		m_data = opGen.GenerateOperand(arguments.at(0), OperandGenerator<B, T>::LoadKind::Vector);
		arguments.at(1)->Accept(*this);

		return m_targetRegister;
	}

	void Visit(const HorseIR::Identifier *identifier) override
	{
		// Generate a loop, iterating over all values and checking if they match
		//
		//   START:
		//      setp %p, ...
		//      @%p br END
		//
		//      <value>
		//      <check>
		//      <increment>
		//
		//      br START
		//
		//   END:

		// Construct the loop

		auto resources = this->m_builder.GetLocalResources();
		auto index = resources->template AllocateTemporary<PTX::UInt32Type>();
		this->m_builder.AddStatement(new PTX::MoveInstruction<PTX::UInt32Type>(index, new PTX::UInt32Value(0)));

		DataSizeGenerator<B> sizeGenerator(this->m_builder);
		auto size = sizeGenerator.GenerateSize(identifier);

		auto startLabel = this->m_builder.CreateLabel("START");
		auto endLabel = this->m_builder.CreateLabel("END");
		auto predicate = resources->template AllocateTemporary<PTX::PredicateType>();

		this->m_builder.AddStatement(startLabel);
		this->m_builder.AddStatement(new PTX::SetPredicateInstruction<PTX::UInt32Type>(predicate, index, size, PTX::UInt32Type::ComparisonOperator::GreaterEqual));
		this->m_builder.AddStatement(new PTX::BranchInstruction(endLabel, predicate));
		this->m_builder.AddStatement(new PTX::BlankStatement());

		// Construct the loop body for checking the next value

		OperandGenerator<B, T> operandGenerator(this->m_builder);
		auto value = operandGenerator.GenerateOperand(identifier, index, this->m_builder.UniqueIdentifier("member"));

		const PTX::Register<PTX::PredicateType> *match = nullptr;
		if constexpr(std::is_same<D, PTX::PredicateType>::value)
		{
			match = m_targetRegister;
		}
		else
		{
			auto resources = this->m_builder.GetLocalResources();
			match = resources->template AllocateTemporary<PTX::PredicateType>();
		}

		// Check the next value

		GenerateMatch(match, value);
		this->m_builder.AddStatement(new PTX::BranchInstruction(endLabel, match));

		// Increment the index by 1

		this->m_builder.AddStatement(new PTX::AddInstruction<PTX::UInt32Type>(index, index, new PTX::UInt32Value(1)));

		// Complete the loop structure. Exit (fallthrough) if match

		this->m_builder.AddStatement(new PTX::BranchInstruction(startLabel));
		this->m_builder.AddStatement(new PTX::BlankStatement());
		this->m_builder.AddStatement(endLabel);

		// If we are computing the index, store the index that exited (== size if not found)

		if constexpr(std::is_same<D, PTX::Int64Type>::value)
		{
			ConversionGenerator::ConvertSource<PTX::Int64Type, PTX::UInt32Type>(this->m_builder, m_targetRegister, index);
		}
	}

	void Visit(const HorseIR::Literal *literal) override
	{
		BuiltinGenerator<B, D>::Unimplemented("literal kind");
	}

	void Visit(const HorseIR::BooleanLiteral *literal) override
	{
		VisitLiteral<std::int8_t>(literal);
	}

	void Visit(const HorseIR::CharLiteral *literal) override
	{
		VisitLiteral<std::int8_t>(literal);
	}

	void Visit(const HorseIR::Int8Literal *literal) override
	{
		VisitLiteral<std::int8_t>(literal);
	}

	void Visit(const HorseIR::Int16Literal *literal) override
	{
		VisitLiteral<std::int16_t>(literal);
	}

	void Visit(const HorseIR::Int32Literal *literal) override
	{
		VisitLiteral<std::int32_t>(literal);
	}

	void Visit(const HorseIR::Int64Literal *literal) override
	{
		VisitLiteral<std::int64_t>(literal);
	}

	void Visit(const HorseIR::Float32Literal *literal) override
	{
		VisitLiteral<float>(literal);
	}

	void Visit(const HorseIR::Float64Literal *literal) override
	{
		VisitLiteral<double>(literal);
	}

	void Visit(const HorseIR::StringLiteral *literal) override
	{
		VisitLiteral<std::string>(literal);
	}

	void Visit(const HorseIR::SymbolLiteral *literal) override
	{
		VisitLiteral<HorseIR::SymbolValue *>(literal);
	}

	void Visit(const HorseIR::DatetimeLiteral *literal) override
	{
		VisitLiteral<HorseIR::DatetimeValue *>(literal);
	}

	void Visit(const HorseIR::MonthLiteral *literal) override
	{
		VisitLiteral<HorseIR::MonthValue *>(literal);
	}

	void Visit(const HorseIR::DateLiteral *literal) override
	{
		VisitLiteral<HorseIR::DateValue *>(literal);
	}

	void Visit(const HorseIR::MinuteLiteral *literal) override
	{
		VisitLiteral<HorseIR::MinuteValue *>(literal);
	}

	void Visit(const HorseIR::SecondLiteral *literal) override
	{
		VisitLiteral<HorseIR::SecondValue *>(literal);
	}

	void Visit(const HorseIR::TimeLiteral *literal) override
	{
		VisitLiteral<HorseIR::TimeValue *>(literal);
	}

	template<class L>
	void VisitLiteral(const HorseIR::TypedVectorLiteral<L> *literal)
	{
		// For each value in the literal, check if it is equal to the data value in this thread. Note, this is an unrolled loop

		for (const auto& value : literal->GetValues())
		{
			// Load the value and cast to the appropriate type

			auto resources = this->m_builder.GetLocalResources();
			auto match = resources->template AllocateTemporary<PTX::PredicateType>();

			if constexpr(std::is_same<L, std::string>::value)
			{
				GenerateMatch(match, new PTX::Value<T>(static_cast<typename T::SystemType>(Runtime::StringBucket::HashString(value))));
			}
			else if constexpr(std::is_same<L, HorseIR::SymbolValue *>::value)
			{
				GenerateMatch(match, new PTX::Value<T>(static_cast<typename T::SystemType>(Runtime::StringBucket::HashString(value->GetName()))));
			}
			else if constexpr(std::is_convertible<L, HorseIR::CalendarValue *>::value)
			{
				GenerateMatch(match, new PTX::Value<T>(static_cast<typename T::SystemType>(value->GetEpochTime())));
			}
			else if constexpr(std::is_convertible<L, HorseIR::ExtendedCalendarValue *>::value)
			{
				GenerateMatch(match, new PTX::Value<T>(static_cast<typename T::SystemType>(value->GetExtendedEpochTime())));
			}
			else
			{
				GenerateMatch(match, new PTX::Value<T>(static_cast<typename T::SystemType>(value)));
			}

			if constexpr(std::is_same<D, PTX::PredicateType>::value)
			{
				this->m_builder.AddStatement(new PTX::OrInstruction<PTX::PredicateType>(m_targetRegister, m_targetRegister, match));
			}
			else
			{
				//TODO: Support @index_of with literals
			}
		}
	}

private:
	void GenerateMatch(const PTX::Register<PTX::PredicateType> *match, const PTX::TypedOperand<T> *value)
	{
		if constexpr(std::is_same<T, PTX::PredicateType>::value)
		{
			auto resources = this->m_builder.GetLocalResources();
			auto temp = resources->template AllocateTemporary<PTX::PredicateType>();

			this->m_builder.AddStatement(new PTX::XorInstruction<PTX::PredicateType>(temp, m_data, value));
			this->m_builder.AddStatement(new PTX::NotInstruction<PTX::PredicateType>(match, temp));
		}
		else
		{
			this->m_builder.AddStatement(new PTX::SetPredicateInstruction<T>(match, m_data, value, T::ComparisonOperator::Equal));
		}
	}

	const PTX::TypedOperand<T> *m_data = nullptr;
	const PTX::Register<D> *m_targetRegister = nullptr;
};

}
