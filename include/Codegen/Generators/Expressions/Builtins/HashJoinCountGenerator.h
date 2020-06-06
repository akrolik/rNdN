#pragma once

#include "Codegen/Generators/Generator.h"

#include "Codegen/Builder.h"
#include "Codegen/Generators/Data/TargetGenerator.h"
#include "Codegen/Generators/Expressions/Builtins/InternalHashGenerator.h"
#include "Codegen/Generators/Indexing/DataIndexGenerator.h"
#include "Codegen/Generators/Indexing/DataSizeGenerator.h"
#include "Codegen/Generators/Indexing/PrefixSumGenerator.h"

#include "HorseIR/Tree/Tree.h"

#include "PTX/PTX.h"

namespace Codegen {

template<PTX::Bits B>
class HashJoinCountGenerator : public Generator
{
public:
	using Generator::Generator;

	std::string Name() const override { return "HashJoinCountGenerator"; }

	void Generate(const std::vector<HorseIR::LValue *>& targets, const std::vector<HorseIR::Operand *>& arguments)
	{
		auto dataArgument = arguments.at(0);
		auto keyArgument = arguments.at(1);

		// Get target registers

		TargetGenerator<B, PTX::Int64Type> targetGenerator(this->m_builder);
		auto offsetsRegister = targetGenerator.Generate(targets.at(0), nullptr);
		auto countRegister = targetGenerator.Generate(targets.at(1), nullptr);

		// Count the number of join results per left-hand data item

		auto resources = this->m_builder.GetLocalResources();

		auto matches = resources->template AllocateTemporary<PTX::Int64Type>();
		this->m_builder.AddStatement(new PTX::MoveInstruction<PTX::Int64Type>(matches, new PTX::Int64Value(0)));

		// Bounds check

		auto startLabel = this->m_builder.CreateLabel("START");
		auto endLabel = this->m_builder.CreateLabel("END");

		DataIndexGenerator<B> indexGenerator(this->m_builder);
		auto index = indexGenerator.GenerateDataIndex();

		DataSizeGenerator<B> sizeGenerator(this->m_builder);
		auto dataSize = sizeGenerator.GenerateSize(dataArgument);
		auto sizePredicate = resources->template AllocateTemporary<PTX::PredicateType>();

		this->m_builder.AddStatement(new PTX::SetPredicateInstruction<PTX::UInt32Type>(sizePredicate, index, dataSize, PTX::UInt32Type::ComparisonOperator::GreaterEqual));
		this->m_builder.AddStatement(new PTX::BranchInstruction(endLabel, sizePredicate));

		auto capacity = sizeGenerator.GenerateSize(keyArgument);
		auto capacityM1 = resources->template AllocateTemporary<PTX::UInt32Type>();
		this->m_builder.AddStatement(new PTX::SubtractInstruction<PTX::UInt32Type>(capacityM1, capacity, new PTX::UInt32Value(1)));

		// Compute the hash

		InternalHashGenerator<B> hashGenerator(this->m_builder);
		auto slot = hashGenerator.Generate(dataArgument);

		// Keep the slot within range

		this->m_builder.AddStatement(startLabel);
		this->m_builder.AddStatement(new PTX::AndInstruction<PTX::Bit32Type>(
			new PTX::Bit32RegisterAdapter<PTX::UIntType>(slot),
			new PTX::Bit32RegisterAdapter<PTX::UIntType>(slot),
			new PTX::Bit32RegisterAdapter<PTX::UIntType>(capacityM1)
		));

		DispatchType(*this, dataArgument->GetType(), matches, slot, dataArgument, keyArgument, startLabel);

		this->m_builder.AddStatement(endLabel);

		// Compute prefix sum, getting the offset for each thread

		PrefixSumGenerator<B, PTX::Int64Type> prefixSumGenerator(this->m_builder);
		auto prefixSum = prefixSumGenerator.Generate(matches, PrefixSumMode::Exclusive);

		// Compute the count of data items

		this->m_builder.AddStatement(new PTX::AddInstruction<PTX::Int64Type>(countRegister, matches, prefixSum));

		// Only store for the last thread

		auto countPredicate = resources->template AllocateTemporary<PTX::PredicateType>();

		auto size = sizeGenerator.GenerateSize(dataArgument);
		auto sizeLess1 = resources->template AllocateTemporary<PTX::UInt32Type>();

		this->m_builder.AddStatement(new PTX::SubtractInstruction<PTX::UInt32Type>(sizeLess1, size, new PTX::UInt32Value(1)));
		this->m_builder.AddStatement(new PTX::SetPredicateInstruction<PTX::UInt32Type>(countPredicate, index, sizeLess1, PTX::UInt32Type::ComparisonOperator::Equal));

		resources->SetCompressedRegister<PTX::Int64Type>(countRegister, countPredicate);
		resources->SetIndexedRegister<PTX::Int64Type>(countRegister, new PTX::UInt32Value(0));

		// Generate output

		this->m_builder.AddStatement(new PTX::MoveInstruction<PTX::Int64Type>(offsetsRegister, prefixSum));
	}

	template<class T>
	void GenerateVector(const PTX::Register<PTX::Int64Type> *matches, const PTX::Register<PTX::UInt32Type> *slot, const HorseIR::Operand *dataOperand, const HorseIR::Operand *keyOperand, const PTX::Label *startLabel)
	{
		GenerateHashLookup<T>(matches, slot, dataOperand, keyOperand, startLabel);
	}

	template<class T>
	void GenerateList(const PTX::Register<PTX::Int64Type> *matches, const PTX::Register<PTX::UInt32Type> *slot, const HorseIR::Operand *dataOperand, const HorseIR::Operand *keyOperand, const PTX::Label *startLabel)
	{
		GenerateHashLookup<T>(matches, slot, dataOperand, keyOperand, startLabel);
	}
	
	template<class T>
	void GenerateTuple(unsigned int index, const PTX::Register<PTX::Int64Type> *matches, const PTX::Register<PTX::UInt32Type> *slot, const HorseIR::Operand *dataOperand, const HorseIR::Operand *keyOperand, const PTX::Label *startLabel)
	{
		if (index == 0)
		{
			GenerateHashLookup<T>(matches, slot, dataOperand, keyOperand, startLabel);
		}
	}

private:
	template<class T>
	void GenerateHashLookup(const PTX::Register<PTX::Int64Type> *matches, const PTX::Register<PTX::UInt32Type> *slot, const HorseIR::Operand *dataOperand, const HorseIR::Operand *keyOperand, const PTX::Label *startLabel)
	{
		if constexpr(std::is_same<T, PTX::PredicateType>::value || std::is_same<T, PTX::Int8Type>::value)
		{
			//TODO: Smaller types
			Error("Unimplemented");
		}
		else
		{
			auto resources = this->m_builder.GetLocalResources();
			auto emptyPredicate = resources->template AllocateTemporary<PTX::PredicateType>();
			auto elseLabel = this->m_builder.CreateLabel("ELSE");

			// Check if we have a match at this slot

			InternalHashEqualGenerator<B, T> equalGenerator(this->m_builder);
			auto [equalPredicate, slotValue] = equalGenerator.Generate(dataOperand, keyOperand, slot);

			this->m_builder.AddStatement(new PTX::AddInstruction<PTX::UInt32Type>(slot, slot, new PTX::UInt32Value(1)));

			this->m_builder.AddStatement(new PTX::BranchInstruction(elseLabel, equalPredicate, true));
			this->m_builder.AddStatement(new PTX::AddInstruction<PTX::Int64Type>(matches, matches, new PTX::Int64Value(1)));
			this->m_builder.AddStatement(new PTX::BranchInstruction(startLabel));

			// Otherwise, check if we have reached the end of the search area

			this->m_builder.AddStatement(elseLabel);

			auto empty = new PTX::Value<T>(std::numeric_limits<typename T::SystemType>::max());
			this->m_builder.AddStatement(new PTX::SetPredicateInstruction<T>(emptyPredicate, slotValue, empty, T::ComparisonOperator::NotEqual));
			this->m_builder.AddStatement(new PTX::BranchInstruction(startLabel, emptyPredicate));
		}
	}
};

}
