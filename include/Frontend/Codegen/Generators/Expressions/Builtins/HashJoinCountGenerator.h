#pragma once

#include "Frontend/Codegen/Generators/Generator.h"

#include "Frontend/Codegen/Builder.h"
#include "Frontend/Codegen/Generators/Data/TargetGenerator.h"
#include "Frontend/Codegen/Generators/Expressions/Builtins/InternalHashGenerator.h"
#include "Frontend/Codegen/Generators/Indexing/DataIndexGenerator.h"
#include "Frontend/Codegen/Generators/Indexing/DataSizeGenerator.h"
#include "Frontend/Codegen/Generators/Indexing/PrefixSumGenerator.h"

#include "HorseIR/Tree/Tree.h"

#include "PTX/Tree/Tree.h"

namespace Frontend {
namespace Codegen {

template<PTX::Bits B>
class HashJoinCountGenerator : public Generator
{
public:
	using Generator::Generator;

	std::string Name() const override { return "HashJoinCountGenerator"; }

	void Generate(const std::vector<const HorseIR::LValue *>& targets, const std::vector<const HorseIR::Operand *>& arguments)
	{
		auto dataArgument = arguments.back();
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
		if constexpr(std::is_same<T, PTX::PredicateType>::value || std::is_same<T, PTX::Int8Type>::value)
		{
			//TODO: Smaller types
			Error("Unimplemented");
		}
		else
		{
			std::vector<const HorseIR::Operand *> functionArguments(std::begin(arguments), std::end(arguments) - 2);
			auto functionCount = functionArguments.size();

			std::vector<ComparisonOperation> joinOperations;
			for (auto functionArgument : functionArguments)
			{
				if (auto functionType = HorseIR::TypeUtils::GetType<HorseIR::FunctionType>(functionArgument->GetType()))
				{
					auto joinOperation = GetJoinComparisonOperation(functionType->GetFunctionDeclaration(), true);
					joinOperations.push_back(joinOperation);
				}
				else
				{
					Generator::Error("non-function join argument '" + HorseIR::PrettyPrinter::PrettyString(functionArgument, true) + "'");
				}
			}

			// Decompose operands

			auto keyOperand = arguments.at(functionCount);
			auto dataOperand = arguments.at(functionCount + 1);

			// Get target registers

			TargetGenerator<B, PTX::Int64Type> targetGenerator(this->m_builder);
			auto offsetsRegister = targetGenerator.Generate(targets.at(0), nullptr);
			auto countRegister = targetGenerator.Generate(targets.at(1), nullptr);

			// Count the number of join results per left-hand data item

			auto resources = this->m_builder.GetLocalResources();

			auto matches = resources->template AllocateTemporary<PTX::Int64Type>();
			this->m_builder.AddStatement(new PTX::MoveInstruction<PTX::Int64Type>(matches, new PTX::Int64Value(0)));

			// Bounds check

			DataIndexGenerator<B> indexGenerator(this->m_builder);
			auto index = indexGenerator.GenerateDataIndex();

			DataSizeGenerator<B> sizeGenerator(this->m_builder);
			auto dataSize = sizeGenerator.GenerateSize(dataOperand);

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
				auto slot = hashGenerator.Generate(dataOperand, joinOperations);

				// Keep the slot within range

				this->m_builder.AddDoWhileLoop("HASH", [&](Builder::LoopContext& loopContext)
				{
					this->m_builder.AddStatement(new PTX::AndInstruction<PTX::Bit32Type>(
						new PTX::Bit32RegisterAdapter<PTX::UIntType>(slot),
						new PTX::Bit32RegisterAdapter<PTX::UIntType>(slot),
						new PTX::Bit32RegisterAdapter<PTX::UIntType>(capacityM1)
					));

					// Check if we have a match at this slot

					InternalHashEqualGenerator<B, T> equalGenerator(this->m_builder);
					auto [equalPredicate, slotValue, slotIdentifier] = equalGenerator.Generate(dataOperand, keyOperand, slot, joinOperations);

					// Count number of matches

					auto matchInstruction = new PTX::AddInstruction<PTX::Int64Type>(matches, matches, new PTX::Int64Value(1));
					matchInstruction->SetPredicate(equalPredicate);
					this->m_builder.AddStatement(matchInstruction);

					// Check if we have reached the end of the search area

					InternalHashEmptyGenerator<B> emptyGenerator(this->m_builder);
					auto emptyPredicate = emptyGenerator.Generate(keyOperand, slot, slotIdentifier);

					// Pre-increment slot position

					this->m_builder.AddStatement(new PTX::AddInstruction<PTX::UInt32Type>(slot, slot, new PTX::UInt32Value(1)));

					return std::make_tuple(emptyPredicate, false);
				});
			});

			// Compute prefix sum, getting the offset for each thread

			PrefixSumGenerator<B, PTX::Int64Type> prefixSumGenerator(this->m_builder);
			auto prefixSum = prefixSumGenerator.Generate(matches, PrefixSumMode::Exclusive);

			// Compute the count of data items

			this->m_builder.AddStatement(new PTX::AddInstruction<PTX::Int64Type>(countRegister, matches, prefixSum));

			// Only store for the last thread

			auto countPredicate = resources->template AllocateTemporary<PTX::PredicateType>();

			auto size = sizeGenerator.GenerateSize(dataOperand);
			auto sizeLess1 = resources->template AllocateTemporary<PTX::UInt32Type>();

			this->m_builder.AddStatement(new PTX::SubtractInstruction<PTX::UInt32Type>(sizeLess1, size, new PTX::UInt32Value(1)));
			this->m_builder.AddStatement(new PTX::SetPredicateInstruction<PTX::UInt32Type>(countPredicate, index, sizeLess1, PTX::UInt32Type::ComparisonOperator::Equal));

			resources->SetCompressedRegister<PTX::Int64Type>(countRegister, countPredicate);
			resources->SetIndexedRegister<PTX::Int64Type>(countRegister, new PTX::UInt32Value(0));

			// Generate output

			this->m_builder.AddStatement(new PTX::MoveInstruction<PTX::Int64Type>(offsetsRegister, prefixSum));
		}
	}
};

}
}
