#pragma once

#include "Codegen/Generators/Generator.h"
#include "HorseIR/Traversal/ConstVisitor.h"

#include "Codegen/Builder.h"
#include "Codegen/Generators/GeometryGenerator.h"
#include "Codegen/Generators/IndexGenerator.h"

#include "HorseIR/Tree/Tree.h"

#include "PTX/PTX.h"

namespace Codegen {

template<PTX::Bits B>
class GroupChangeGenerator : public Generator, public HorseIR::ConstVisitor
{
public:
	using Generator::Generator;

	void Generate(const std::vector<HorseIR::Operand *>& dataArguments, const PTX::TypedOperand<PTX::UInt32Type> *index, const PTX::Register<PTX::PredicateType> *predicate)
	{
		m_predicate = predicate;
		m_index = index;
		for (const auto argument : dataArguments)
		{
			argument->Accept(*this);
		}
	}

	void Visit(const HorseIR::Identifier *identifier) override
	{
		DispatchType(*this, identifier->GetType(), identifier);
	}

	template<class T>
	void Generate(const HorseIR::Identifier *identifier)
	{
		//TODO: Use comparison generator instead to generate optimized comparison
		if constexpr(std::is_same<T, PTX::PredicateType>::value || std::is_same<T, PTX::Int8Type>::value)
		{
			Generate<PTX::Int16Type>(identifier);
		}
		else
		{
			auto resources = this->m_builder.GetLocalResources();

			// Load the current value

			OperandGenerator<B, T> opGen(this->m_builder);
			auto value = opGen.GenerateOperand(identifier, m_index, "val");

			// Load the previous value, duplicating the first element (as there is no previous element)

			auto indexM1 = resources->template AllocateTemporary<PTX::UInt32Type>();
			this->m_builder.AddStatement(new PTX::SubtractInstruction<PTX::UInt32Type>(indexM1, m_index, new PTX::UInt32Value(1)));

			auto previousPredicate = resources->template AllocateTemporary<PTX::PredicateType>();
			this->m_builder.AddStatement(new PTX::SetPredicateInstruction<PTX::UInt32Type>(
				previousPredicate, m_index, new PTX::UInt32Value(0), PTX::UInt32Type::ComparisonOperator::NotEqual
			)); 

			auto indexPrevious = resources->template AllocateTemporary<PTX::UInt32Type>();
			this->m_builder.AddStatement(new PTX::SelectInstruction<PTX::UInt32Type>(indexPrevious, indexM1, m_index, previousPredicate));

			auto previousValue = opGen.GenerateOperand(identifier, indexPrevious, "prev");

			// Check if the value is different

			this->m_builder.AddStatement(new PTX::SetPredicateInstruction<T>(
				m_predicate, nullptr, value, previousValue, T::ComparisonOperator::NotEqual, m_predicate, PTX::PredicateModifier::BoolOperator::Or
			)); 
		}
	}

private:
	const PTX::TypedOperand<PTX::UInt32Type> *m_index = nullptr;
	const PTX::Register<PTX::PredicateType> *m_predicate = nullptr;
};

template<PTX::Bits B>
class GroupGenerator : public Generator
{
public:
	using Generator::Generator;

	void Generate(const std::vector<HorseIR::LValue *>& targets, const std::vector<HorseIR::Operand *>& arguments)
	{
		auto resources = this->m_builder.GetLocalResources();

		// Convenience split of input arguments

		std::vector<HorseIR::Operand *> dataArguments(std::begin(arguments) + 1, std::end(arguments));
		const auto indexArgument = arguments.at(0);

		// Initialize the current and previous values, and compute the change
		//   
		//   1. Check if size is within bounds
		//   2. Load the current value
		//   3. Load the previous value at index -1 (bounded below by index 0)

		// Check the size is within bounds, if so we will load both values and do a comparison

		IndexGenerator indexGenerator(this->m_builder);
		auto index = indexGenerator.GenerateGlobalIndex();

		auto changePredicate = resources->template AllocateTemporary<PTX::PredicateType>();
		this->m_builder.AddStatement(new PTX::MoveInstruction<PTX::PredicateType>(changePredicate, new PTX::BoolValue(0)));

		GeometryGenerator geometryGenerator(this->m_builder);
		auto size = geometryGenerator.GenerateVectorSize();

		auto sizeLabel = this->m_builder.CreateLabel("SIZE");
		auto sizePredicate = resources->template AllocateTemporary<PTX::PredicateType>();

		this->m_builder.AddStatement(new PTX::SetPredicateInstruction<PTX::UInt32Type>(sizePredicate, index, size, PTX::UInt32Type::ComparisonOperator::GreaterEqual));
		this->m_builder.AddStatement(new PTX::BranchInstruction(sizeLabel, sizePredicate));

		// Check for each column if there has been a change

		GroupChangeGenerator<B> changeGenerator(this->m_builder);
		changeGenerator.Generate(dataArguments, index, changePredicate);

		// Completed determining size

		this->m_builder.AddStatement(new PTX::BlankStatement());
		this->m_builder.AddStatement(sizeLabel);

		// Get the return targets!

		TargetGenerator<B, PTX::Int64Type> targetGenerator(this->m_builder);
		auto keys = targetGenerator.Generate(targets.at(0), nullptr);
		auto values = targetGenerator.Generate(targets.at(1), nullptr);

		// Set the key as the dataIndex value

		auto compressPredicate = resources->template AllocateTemporary<PTX::PredicateType>();
		this->m_builder.AddStatement(new PTX::SetPredicateInstruction<PTX::UInt32Type>(
			compressPredicate, nullptr, index, new PTX::UInt32Value(0), PTX::UInt32Type::ComparisonOperator::Equal, changePredicate, PTX::PredicateModifier::BoolOperator::Or
		));

		OperandGenerator<B, PTX::Int64Type> operandGenerator(this->m_builder);
		auto dataIndex = operandGenerator.GenerateOperand(indexArgument, index, this->m_builder.UniqueIdentifier("index"));

		this->m_builder.AddStatement(new PTX::MoveInstruction<PTX::Int64Type>(keys, dataIndex));

		resources->SetCompressedRegister(keys, compressPredicate);

		// Set the value as the index into the dataIndex

		ConversionGenerator::ConvertSource(this->m_builder, values, index);
		resources->SetCompressedRegister(values, compressPredicate);
	}
};

}
