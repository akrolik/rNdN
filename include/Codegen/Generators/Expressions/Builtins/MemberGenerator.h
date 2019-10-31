#pragma once

#include "HorseIR/Traversal/ConstVisitor.h"
#include "Codegen/Generators/Expressions/Builtins/BuiltinGenerator.h"

#include "Codegen/Builder.h"
#include "Codegen/Generators/Expressions/OperandCompressionGenerator.h"
#include "Codegen/Generators/Expressions/OperandGenerator.h"

#include "HorseIR/Tree/Tree.h"

#include "PTX/PTX.h"

namespace Codegen {

template<PTX::Bits B, class T>
class MemeberGeneratorInternal : public BuiltinGenerator<B, PTX::PredicateType>, public HorseIR::ConstVisitor
{
public:
	using BuiltinGenerator<B, PTX::PredicateType>::BuiltinGenerator;

	const PTX::Register<PTX::PredicateType> *Generate(const HorseIR::LValue *target, const std::vector<HorseIR::Operand *>& arguments)
	{
		m_target = target;
		m_arguments = arguments;

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

		auto resources = this->m_builder.GetLocalResources();

		m_targetRegister = this->GenerateTargetRegister(m_target, m_arguments);
		this->m_builder.AddStatement(new PTX::MoveInstruction<PTX::PredicateType>(m_targetRegister, new PTX::BoolValue(false)));

		auto index = resources->template AllocateTemporary<PTX::UInt32Type>();
		this->m_builder.AddStatement(new PTX::MoveInstruction<PTX::UInt32Type>(index, new PTX::UInt32Value(0)));

		SizeGenerator<B> sizeGenerator(this->m_builder);
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

		GenerateMerge(value);

		// Increment the index by 1

		this->m_builder.AddStatement(new PTX::AddInstruction<PTX::UInt32Type>(index, index, new PTX::UInt32Value(1)));

		// Complete the loop structure

		this->m_builder.AddStatement(new PTX::BranchInstruction(startLabel));
		this->m_builder.AddStatement(new PTX::BlankStatement());
		this->m_builder.AddStatement(endLabel);
	}

	void Visit(const HorseIR::Literal *literal) override
	{
		BuiltinGenerator<B, PTX::PredicateType>::Unimplemented("literal kind");
	}

	void Visit(const HorseIR::BooleanLiteral *literal) override
	{
		Generate<char>(literal);
	}

	void Visit(const HorseIR::CharLiteral *literal) override
	{
		Generate<char>(literal);
	}

	void Visit(const HorseIR::Int8Literal *literal) override
	{
		Generate<std::int8_t>(literal);
	}

	void Visit(const HorseIR::Int16Literal *literal) override
	{
		Generate<std::int16_t>(literal);
	}

	void Visit(const HorseIR::Int32Literal *literal) override
	{
		Generate<std::int32_t>(literal);
	}

	void Visit(const HorseIR::Int64Literal *literal) override
	{
		Generate<std::int64_t>(literal);
	}

	void Visit(const HorseIR::Float32Literal *literal) override
	{
		Generate<float>(literal);
	}

	void Visit(const HorseIR::Float64Literal *literal) override
	{
		Generate<double>(literal);
	}

	template<class L>
	void Generate(const HorseIR::TypedVectorLiteral<L> *literal)
	{
		// Generate a zero'd target register

		m_targetRegister = this->GenerateTargetRegister(m_target, m_arguments);
		this->m_builder.AddStatement(new PTX::MoveInstruction<PTX::PredicateType>(m_targetRegister, new PTX::BoolValue(false)));

		// For each value in the literal, check if it is equal to the data value in this thread. Note, this is an unrolled loop

		for (auto literalValue : literal->GetValues())
		{
			// Load the value and cast to the appropriate type

			if constexpr(std::is_same<typename T::SystemType, L>::value)
			{
				GenerateMerge(new PTX::Value<T>(literalValue));
			}
			else
			{
				GenerateMerge(new PTX::Value<T>(static_cast<typename T::SystemType>(literalValue)));
			}
		}
	}

private:
	void GenerateMerge(const PTX::TypedOperand<T> *value)
	{
		// Check and merge (or) with the previous result

		if constexpr(std::is_same<T, PTX::PredicateType>::value)
		{
			auto resources = this->m_builder.GetLocalResources();
			auto temp = resources->template AllocateTemporary<PTX::PredicateType>();

			this->m_builder.AddStatement(new PTX::XorInstruction<PTX::PredicateType>(temp, m_data, value));
			this->m_builder.AddStatement(new PTX::NotInstruction<PTX::PredicateType>(temp, temp));
			this->m_builder.AddStatement(new PTX::OrInstruction<PTX::PredicateType>(m_targetRegister, m_targetRegister, temp));
		}
		else
		{
			this->m_builder.AddStatement(
				new PTX::SetPredicateInstruction<T>(m_targetRegister, nullptr, m_data, value, T::ComparisonOperator::Equal, m_targetRegister, PTX::PredicateModifier::BoolOperator::Or)
			);
		}
	}

	const PTX::TypedOperand<T> *m_data = nullptr;
	const PTX::Register<PTX::PredicateType> *m_targetRegister = nullptr;

	const HorseIR::LValue *m_target = nullptr;
	std::vector<HorseIR::Operand *> m_arguments;
};

template<PTX::Bits B, class T>
class MemberGenerator : public BuiltinGenerator<B, T>
{
public: 
	using BuiltinGenerator<B, T>::BuiltinGenerator;
};

template<PTX::Bits B>
class MemberGenerator<B, PTX::PredicateType>: public BuiltinGenerator<B, PTX::PredicateType>
{
public:
	using BuiltinGenerator<B, PTX::PredicateType>::BuiltinGenerator;

	const PTX::Register<PTX::PredicateType> *GenerateCompressionPredicate(const std::vector<HorseIR::Operand *>& arguments) override
	{
		return OperandCompressionGenerator::UnaryCompressionRegister(this->m_builder, arguments);
	}

	const PTX::Register<PTX::PredicateType> *Generate(const HorseIR::LValue *target, const std::vector<HorseIR::Operand *>& arguments) override
	{
		DispatchType(*this, arguments.at(0)->GetType(), target, arguments);
		return m_targetRegister;
	}

	template<typename T>
	void Generate(const HorseIR::LValue *target, const std::vector<HorseIR::Operand *>& arguments)
	{
		if constexpr(std::is_same<T, PTX::Int8Type>::value)
		{
			// Comparison can only occur for 16-bits+

			Generate<PTX::Int16Type>(target, arguments);
		}
		else
		{
			MemeberGeneratorInternal<B, T> memberGenerator(this->m_builder);
			m_targetRegister = memberGenerator.Generate(target, arguments);
		}
	}

private:
	const PTX::Register<PTX::PredicateType> *m_targetRegister = nullptr;
};

}
