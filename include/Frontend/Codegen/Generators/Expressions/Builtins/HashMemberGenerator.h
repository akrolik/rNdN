#pragma once

#include "Frontend/Codegen/Generators/Generator.h"

#include "Frontend/Codegen/Builder.h"
#include "Frontend/Codegen/Generators/Data/TargetGenerator.h"
#include "Frontend/Codegen/Generators/Expressions/Builtins/InternalHashGenerator.h"
#include "Frontend/Codegen/Generators/Indexing/DataIndexGenerator.h"
#include "Frontend/Codegen/Generators/Indexing/DataSizeGenerator.h"

#include "HorseIR/Tree/Tree.h"

#include "PTX/Tree/Tree.h"

namespace Frontend {
namespace Codegen {

template<PTX::Bits B>
class HashMemberGenerator : public Generator
{
public:
	using Generator::Generator;

	std::string Name() const override { return "HashMemberGenerator"; }

	void Generate(const std::vector<const HorseIR::LValue *>& targets, const std::vector<const HorseIR::Operand *>& arguments)
	{
		auto dataArgument = arguments.at(1);
		DispatchType(*this, dataArgument->GetType(), targets, arguments);
	}

	template<class T>
	void GenerateVector(const std::vector<const HorseIR::LValue *>& targets, const std::vector<const HorseIR::Operand *>& arguments)
	{
		GenerateHashLookup<T>(targets, arguments);
	}

	template<class T>
	void GenerateList(const std::vector<const HorseIR::LValue *>& targets, const std::vector<const HorseIR::Operand *>& arguments)
	{
		GenerateHashLookup<T>(targets, arguments);
	}
	
	template<class T>
	void GenerateTuple(unsigned int index, const std::vector<const HorseIR::LValue *>& targets, const std::vector<const HorseIR::Operand *>& arguments)
	{
		if (index == 0)
		{
			GenerateHashLookup<T>(targets, arguments);
		}
	}

private:
	template<class T>
	void GenerateHashLookup(const std::vector<const HorseIR::LValue *>& targets, const std::vector<const HorseIR::Operand *>& arguments)
	{
		// Decompose operands

		auto keyOperand = arguments.at(0);
		auto dataOperand = arguments.at(1);

		// Get target register

		TargetGenerator<B, PTX::PredicateType> targetGenerator(this->m_builder);
		auto matchPredicate = targetGenerator.Generate(targets.at(0), nullptr);

		// Check for at least one match per left-hand data item

		this->m_builder.AddStatement(new PTX::MoveInstruction<PTX::PredicateType>(matchPredicate, new PTX::BoolValue(false)));

		// Bounds check

		DataIndexGenerator<B> indexGenerator(this->m_builder);
		auto index = indexGenerator.GenerateDataIndex();

		DataSizeGenerator<B> sizeGenerator(this->m_builder);
		auto dataSize = sizeGenerator.GenerateSize(dataOperand);

		auto resources = this->m_builder.GetLocalResources();

		this->m_builder.AddIfStatement("HASH_SKIP", [&]()
		{
			auto sizePredicate = resources->template AllocateTemporary<PTX::PredicateType>();
			this->m_builder.AddStatement(new PTX::SetPredicateInstruction<PTX::UInt32Type>(
				sizePredicate, index, dataSize, PTX::UInt32Type::ComparisonOperator::GreaterEqual
			));
			return std::make_tuple(sizePredicate, false);
		},
		[&]()
		{
			auto capacity = sizeGenerator.GenerateSize(keyOperand);
			auto capacityM1 = resources->template AllocateTemporary<PTX::UInt32Type>();
			this->m_builder.AddStatement(new PTX::SubtractInstruction<PTX::UInt32Type>(capacityM1, capacity, new PTX::UInt32Value(1)));

			// Compute the hash

			InternalHashGenerator<B> hashGenerator(this->m_builder);
			auto slot = hashGenerator.Generate(dataOperand, {ComparisonOperation::Equal});

			// Keep the slot within range

			this->m_builder.AddDoWhileLoop("HASH", [&](Builder::LoopContext& loopContext)
			{
				this->m_builder.AddStatement(new PTX::AndInstruction<PTX::Bit32Type>(
					new PTX::Bit32RegisterAdapter<PTX::UIntType>(slot),
					new PTX::Bit32RegisterAdapter<PTX::UIntType>(slot),
					new PTX::Bit32RegisterAdapter<PTX::UIntType>(capacityM1)
				));

				// Check if we have a match at this slot, break if so

				InternalHashEqualGenerator<B, T> equalGenerator(this->m_builder);
				auto [equalPredicate, slotValue, slotIdentifier] = equalGenerator.Generate(dataOperand, keyOperand, slot, {ComparisonOperation::Equal});
				
				this->m_builder.AddStatement(new PTX::MoveInstruction<PTX::PredicateType>(matchPredicate, equalPredicate));
				this->m_builder.AddBreakStatement(loopContext, matchPredicate);

				// Check if we have reached the end of the search area

				InternalHashEmptyGenerator<B> emptyGenerator(this->m_builder);
				auto emptyPredicate = emptyGenerator.Generate(keyOperand, slot, slotIdentifier);

				// Pre-increment slot position

				this->m_builder.AddStatement(new PTX::AddInstruction<PTX::UInt32Type>(slot, slot, new PTX::UInt32Value(1)));

				return std::make_tuple(emptyPredicate, false);
			});
		});
	}
};

}
}
