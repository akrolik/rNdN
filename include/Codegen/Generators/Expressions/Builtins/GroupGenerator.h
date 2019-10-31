#pragma once

#include "Codegen/Generators/Generator.h"
#include "HorseIR/Traversal/ConstVisitor.h"

#include "Codegen/Generators/AddressGenerator.h"
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

	const PTX::Register<PTX::PredicateType> *Generate(const std::vector<HorseIR::Operand *>& dataArguments, const PTX::TypedOperand<PTX::UInt32Type> *index)
	{
		m_predicate = nullptr;
		m_index = index;
		for (const auto argument : dataArguments)
		{
			argument->Accept(*this);
		}
		return m_predicate;
	}

	void Visit(const HorseIR::Identifier *identifier) override
	{
		DispatchType(*this, identifier->GetType(), identifier);
	}

	template<class T>
	void Generate(const HorseIR::Identifier *identifier)
	{
		if constexpr(std::is_same<T, PTX::PredicateType>::value|| std::is_same<T, PTX::Int8Type>::value)
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
			this->m_builder.AddStatement(new PTX::SetPredicateInstruction<PTX::UInt32Type>(previousPredicate, m_index, new PTX::UInt32Value(0), PTX::UInt32Type::ComparisonOperator::NotEqual)); 

			auto indexPrevious = resources->template AllocateTemporary<PTX::UInt32Type>();
			this->m_builder.AddStatement(new PTX::SelectInstruction<PTX::UInt32Type>(indexPrevious, indexM1, m_index, previousPredicate));

			auto previousValue = opGen.GenerateOperand(identifier, indexPrevious, "prev");

			// Check if the value is different

			auto changePredicate = resources->template AllocateTemporary<PTX::PredicateType>();
			this->m_builder.AddStatement(new PTX::SetPredicateInstruction<T>(changePredicate, value, previousValue, T::ComparisonOperator::NotEqual)); 

			if (m_predicate == nullptr)
			{
				m_predicate = changePredicate;
			}
			else
			{
				this->m_builder.AddStatement(new PTX::OrInstruction<PTX::PredicateType>(m_predicate, m_predicate, changePredicate));
			}
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

	void Generate(const std::vector<HorseIR::LValue *>&targets, const std::vector<HorseIR::Operand *>& arguments)
	{
		// A global prefix sum is computed in 4 stages:
		//
		//    1. Each warp computes its local prefix sum using a shuffle reduction
		//    2. Each block computes its block prefix sum using shared memory and a single warp
		//       propagation step
		//    3. Blocks proceed 1-by-1 and increment a global prefix sum, fetching the previous
		//    4. The local prefix sum is computed using the fetched global value
		//
		// Note that this requires we process all previous blocks beforehand. Since on the GPU
		// we have no guarantee of thread-block ordering, use a global variable to allocate
		// blocks on an arrival basis

		std::vector<HorseIR::Operand *> dataArguments(std::begin(arguments) + 1, std::end(arguments));
		const auto indexArgument = arguments.at(0);

		auto resources = this->m_builder.GetLocalResources();
		auto kernelResources = this->m_builder.GetKernelResources();
		auto globalResources = this->m_builder.GetGlobalResources();

		auto& targetOptions = this->m_builder.GetTargetOptions();

		IndexGenerator indexGenerator(this->m_builder);
		GeometryGenerator geometryGenerator(this->m_builder);
		AddressGenerator<B> addressGenerator(this->m_builder);

		// Allocate global counters and prefix sum

		auto g_initBlocks = globalResources->template AllocateGlobalVariable<PTX::UInt32Type>("initBlocks");
		auto g_completedBlocks = globalResources->template AllocateGlobalVariable<PTX::UInt32Type>("completedBlocks");
		auto g_prefixSum = globalResources->template AllocateGlobalVariable<PTX::UInt32Type>("prefixSum");

		// Initialize the unique block id

		auto s_blockIndex = kernelResources->template AllocateSharedVariable<PTX::UInt32Type>("blockIndex");
		auto blockIndex = resources->template AllocateTemporary<PTX::UInt32Type>();
		auto blockIndexLabel = this->m_builder.CreateLabel("BLOCK_INDEX");

		auto localIndex = indexGenerator.GenerateLocalIndex();
		auto local0Predicate = resources->template AllocateTemporary<PTX::PredicateType>();

		this->m_builder.AddStatement(new PTX::SetPredicateInstruction<PTX::UInt32Type>(local0Predicate, localIndex, new PTX::UInt32Value(0), PTX::UInt32Type::ComparisonOperator::NotEqual));
		this->m_builder.AddStatement(new PTX::BranchInstruction(blockIndexLabel, local0Predicate));

		auto g_initBlocksAddress = addressGenerator.GenerateAddress(g_initBlocks, nullptr);
		this->m_builder.AddStatement(new PTX::AtomicInstruction<B, PTX::UInt32Type, PTX::GlobalSpace, PTX::UInt32Type::AtomicOperation::Add>(
					blockIndex, g_initBlocksAddress, new PTX::UInt32Value(1)
					));

		auto s_blockIndexAddress = addressGenerator.GenerateAddress(s_blockIndex, nullptr);
		this->m_builder.AddStatement(new PTX::StoreInstruction<B, PTX::UInt32Type, PTX::SharedSpace>(s_blockIndexAddress, blockIndex));

		this->m_builder.AddStatement(blockIndexLabel);
		this->m_builder.AddStatement(new PTX::BlankStatement());
		this->m_builder.AddStatement(new PTX::BarrierInstruction(new PTX::UInt32Value(0)));

		// Generate the index using the custom block id

		this->m_builder.AddStatement(new PTX::LoadInstruction<B, PTX::UInt32Type, PTX::SharedSpace>(blockIndex, s_blockIndexAddress));
		auto index = indexGenerator.GenerateGlobalIndex(blockIndex);

		// Initialize the current and previous values, and compute the change
		//   
		//   1. Check if size is within bounds
		//   2. Load the current value
		//   3. Load the previous value at index -1 (bounded below by index 0)

		// Check the size is within bounds, if so we will load both values and do a comparison

		auto change = resources->template AllocateTemporary<PTX::UInt32Type>();
		this->m_builder.AddStatement(new PTX::MoveInstruction<PTX::UInt32Type>(change, new PTX::UInt32Value(0)));

		auto size = geometryGenerator.GenerateVectorSize();
		auto sizePredicate = resources->template AllocateTemporary<PTX::PredicateType>();

		auto sizeLabel = this->m_builder.CreateLabel("SIZE");

		this->m_builder.AddStatement(new PTX::SetPredicateInstruction<PTX::UInt32Type>(sizePredicate, index, size, PTX::UInt32Type::ComparisonOperator::GreaterEqual));
		this->m_builder.AddStatement(new PTX::BranchInstruction(sizeLabel, sizePredicate));

		// Check for each column if there has been a change

		GroupChangeGenerator<B> changeGenerator(this->m_builder);
		auto changePredicate = changeGenerator.Generate(dataArguments, index);

		this->m_builder.AddStatement(new PTX::SelectInstruction<PTX::UInt32Type>(change, new PTX::UInt32Value(1), new PTX::UInt32Value(0), changePredicate));

		// Completed determining size

		this->m_builder.AddStatement(sizeLabel);
		this->m_builder.AddStatement(new PTX::BlankStatement());

		auto changes = resources->template AllocateTemporary<PTX::UInt32Type>();
		this->m_builder.AddStatement(new PTX::MoveInstruction<PTX::UInt32Type>(changes, change));

		// Iteratively compute the prefix sum for the warp

		auto laneIndex = indexGenerator.GenerateLaneIndex();
		GenerateWarpPrefixSum(laneIndex, changes);

		// For each warp, store the last (laneIndex == 31) value in shared memory at its warp index. Adapt from array to variable for addressing

		auto s_prefixSum = new PTX::ArrayVariableAdapter<PTX::UInt32Type, 32, PTX::SharedSpace>(
			kernelResources->template AllocateSharedVariable<PTX::ArrayType<PTX::UInt32Type, 32>>("warpPrefixSums")
		);

		auto warpIndex = indexGenerator.GenerateWarpIndex();
		auto s_prefixSumWarpAddress = addressGenerator.GenerateAddress(s_prefixSum, warpIndex);

		auto warpStoreLabel = this->m_builder.CreateLabel("WARP_STORE");
		auto lastLanePredicate = resources->template AllocateTemporary<PTX::PredicateType>();

		this->m_builder.AddStatement(new PTX::SetPredicateInstruction<PTX::UInt32Type>(
			lastLanePredicate, laneIndex, new PTX::UInt32Value(targetOptions.WarpSize - 1), PTX::UInt32Type::ComparisonOperator::NotEqual
		));
		this->m_builder.AddStatement(new PTX::BranchInstruction(warpStoreLabel, lastLanePredicate));
		this->m_builder.AddStatement(new PTX::StoreInstruction<B, PTX::UInt32Type, PTX::SharedSpace>(s_prefixSumWarpAddress, changes));

		// Synchronize the result so all values are visible to the first warp

		this->m_builder.AddStatement(warpStoreLabel);
		this->m_builder.AddStatement(new PTX::BlankStatement());
		this->m_builder.AddStatement(new PTX::BarrierInstruction(new PTX::UInt32Value(0)));

		// In the first warp, load the values back into a new register and prefix sum

		auto firstWarpPredicate = resources->template AllocateTemporary<PTX::PredicateType>();
		this->m_builder.AddStatement(new PTX::SetPredicateInstruction<PTX::UInt32Type>(
			firstWarpPredicate, warpIndex, new PTX::UInt32Value(0), PTX::UInt32Type::ComparisonOperator::NotEqual
		));

		auto blockSumLabel = this->m_builder.CreateLabel("BLOCK_SUM");
		this->m_builder.AddStatement(new PTX::BranchInstruction(blockSumLabel, firstWarpPredicate));

		// Compute the address for each lane in the first warp, and load the value

		auto s_prefixSumLaneAddress = addressGenerator.GenerateAddress(s_prefixSum, laneIndex);

		auto warpChanges = resources->template AllocateTemporary<PTX::UInt32Type>();
		this->m_builder.AddStatement(new PTX::LoadInstruction<B, PTX::UInt32Type, PTX::SharedSpace>(warpChanges, s_prefixSumLaneAddress));

		// Prefix sum on the first warp

		GenerateWarpPrefixSum(laneIndex, warpChanges);

		// Store the value back and synchronize between all warps

		this->m_builder.AddStatement(new PTX::StoreInstruction<B, PTX::UInt32Type, PTX::SharedSpace>(s_prefixSumLaneAddress, warpChanges));

		this->m_builder.AddStatement(blockSumLabel);
		this->m_builder.AddStatement(new PTX::BlankStatement());
		this->m_builder.AddStatement(new PTX::BarrierInstruction(new PTX::UInt32Value(0)));

		// If we are not the first warp, add the summed result from the shared memory - this completes the prefix sum for the block (!predicate)

		auto warpRestoreLabel = this->m_builder.CreateLabel("WARP_RESTORE");
		this->m_builder.AddStatement(new PTX::BranchInstruction(warpRestoreLabel, firstWarpPredicate, true));

		// Get the prefix sum (inclusive) from the previous warp

		auto warpPrefixSum = resources->template AllocateTemporary<PTX::UInt32Type>();
		auto s_prefixSumWarpAddressM1 = s_prefixSumWarpAddress->CreateOffsetAddress(-1);
		this->m_builder.AddStatement(new PTX::LoadInstruction<B, PTX::UInt32Type, PTX::SharedSpace>(warpPrefixSum, s_prefixSumWarpAddressM1));

		// Add to each value within the warp - computing the block local prefix sum

		this->m_builder.AddStatement(new PTX::AddInstruction<PTX::UInt32Type>(changes, changes, warpPrefixSum));
		this->m_builder.AddStatement(warpRestoreLabel);
		this->m_builder.AddStatement(new PTX::BlankStatement());

		// For each block, load the previous block's value once it is completed. This forms a linear chain, but is fairly efficient

		auto blockPrefixSum = resources->template AllocateTemporary<PTX::UInt32Type>();
		auto s_blockPrefixSum = kernelResources->template AllocateSharedVariable<PTX::UInt32Type>("blockPefixSum");
		auto s_blockPrefixSumAddress = addressGenerator.GenerateAddress(s_blockPrefixSum, nullptr);

		// To join loading/storing into a single section, the LAST thread of a block is responsible for loading/storing

		auto ntidx = resources->template AllocateTemporary<PTX::UInt32Type>();
		auto srntidx = new PTX::IndexedRegister4<PTX::UInt32Type>(PTX::SpecialRegisterDeclaration_ntid->GetVariable("%ntid"), PTX::VectorElement::X);
		this->m_builder.AddStatement(new PTX::MoveInstruction<PTX::UInt32Type>(ntidx, srntidx));

		auto lastLocalIndex = resources->template AllocateTemporary<PTX::UInt32Type>();
		this->m_builder.AddStatement(new PTX::SubtractInstruction<PTX::UInt32Type>(lastLocalIndex, ntidx, new PTX::UInt32Value(1)));

		auto propagateLabel = this->m_builder.CreateLabel("PROPAGATE");
		auto propagatePredicate = resources->template AllocateTemporary<PTX::PredicateType>();
		this->m_builder.AddStatement(new PTX::SetPredicateInstruction<PTX::UInt32Type>(
			propagatePredicate, localIndex, lastLocalIndex, PTX::UInt32Type::ComparisonOperator::NotEqual
		));
		this->m_builder.AddStatement(new PTX::BranchInstruction(propagateLabel, propagatePredicate));

		// Atomically check if the previous block has completed by checking a global counter

		auto atomicStartLabel = this->m_builder.CreateLabel("START");
		this->m_builder.AddStatement(atomicStartLabel);
		this->m_builder.AddStatement(new PTX::BlankStatement());

		auto completedBlocks = resources->template AllocateTemporary<PTX::UInt32Type>();
		auto g_completedBlocksAddress = addressGenerator.GenerateAddress(g_completedBlocks, nullptr);

		// Get the current value by incrementing by 0

		this->m_builder.AddStatement(new PTX::AtomicInstruction<B, PTX::UInt32Type, PTX::GlobalSpace, PTX::UInt32Type::AtomicOperation::Add>(
			completedBlocks, g_completedBlocksAddress, new PTX::UInt32Value(0)
		));

		// Check if we are next, or keep looping!
		
		auto atomicPredicate = resources->template AllocateTemporary<PTX::PredicateType>();
		this->m_builder.AddStatement(new PTX::SetPredicateInstruction<PTX::UInt32Type>(
			atomicPredicate, completedBlocks, blockIndex, PTX::UInt32Type::ComparisonOperator::Less
		));
		this->m_builder.AddStatement(new PTX::BranchInstruction(atomicStartLabel, atomicPredicate));

		// Get the prefix sum up to and including the previous block, and store for all in the block. This is the GLOBAL prefix sum up to this point!

		auto g_prefixSumAddress = addressGenerator.GenerateAddress(g_prefixSum, nullptr);
		this->m_builder.AddStatement(new PTX::LoadInstruction<B, PTX::UInt32Type, PTX::GlobalSpace>(blockPrefixSum, g_prefixSumAddress));
		this->m_builder.AddStatement(new PTX::StoreInstruction<B, PTX::UInt32Type, PTX::SharedSpace>(s_blockPrefixSumAddress, blockPrefixSum));

		// Update the global prefix sum to include this block

		this->m_builder.AddStatement(new PTX::ReductionInstruction<B, PTX::UInt32Type, PTX::GlobalSpace, PTX::UInt32Type::AtomicOperation::Add>(
			g_prefixSumAddress, changes
		));

		// Proceed to next thread by incrementing the global counter

		this->m_builder.AddStatement(new PTX::AddInstruction<PTX::UInt32Type>(completedBlocks, completedBlocks, new PTX::UInt32Value(1)));
		this->m_builder.AddStatement(new PTX::StoreInstruction<B, PTX::UInt32Type, PTX::GlobalSpace>(g_completedBlocksAddress, completedBlocks));

		this->m_builder.AddStatement(propagateLabel);
		this->m_builder.AddStatement(new PTX::BlankStatement());

		// Synchronize the results - every thread now has the previous thread's (inclusive) prefix sum

		this->m_builder.AddStatement(new PTX::BarrierInstruction(new PTX::UInt32Value(0)));

		// Load the prefix sum from the shared memory, and add to the current changes count. This is the GLOBAL prefix sum!

		this->m_builder.AddStatement(new PTX::LoadInstruction<B, PTX::UInt32Type, PTX::SharedSpace>(blockPrefixSum, s_blockPrefixSumAddress));
		this->m_builder.AddStatement(new PTX::AddInstruction<PTX::UInt32Type>(changes, blockPrefixSum, changes));

		// Get the return targets!

		TargetGenerator<B, PTX::Int64Type> targetGenerator(this->m_builder);
		auto keys = targetGenerator.Generate(targets.at(0), nullptr);
		auto values = targetGenerator.Generate(targets.at(1), nullptr);

		// Set the key as the dataIndex value

		OperandGenerator<B, PTX::Int64Type> operandGenerator(this->m_builder);
		auto dataIndex = operandGenerator.GenerateOperand(indexArgument, index, "index");

		this->m_builder.AddStatement(new PTX::MoveInstruction<PTX::Int64Type>(keys, dataIndex));

		auto compressPredicate = resources->template AllocateTemporary<PTX::PredicateType>();
		this->m_builder.AddStatement(new PTX::SetPredicateInstruction<PTX::UInt32Type>(
			compressPredicate, nullptr, index, new PTX::UInt32Value(0), PTX::UInt32Type::ComparisonOperator::Equal, changePredicate, PTX::PredicateModifier::BoolOperator::Or
		));

		resources->SetCompressedRegister(keys, compressPredicate);
		resources->SetIndexedRegister(keys, changes);

		// Set the value as the index into the dataIndex

		ConversionGenerator::ConvertSource(this->m_builder, values, index);

		resources->SetCompressedRegister(values, changePredicate);
		resources->SetIndexedRegister(values, changes);

		// Generate the output size from the final thread prefix sum

		auto nctaidx = resources->template AllocateTemporary<PTX::UInt32Type>();
		auto srnctaidx = new PTX::IndexedRegister4<PTX::UInt32Type>(PTX::SpecialRegisterDeclaration_nctaid->GetVariable("%nctaid"), PTX::VectorElement::X);

		this->m_builder.AddStatement(new PTX::MoveInstruction<PTX::UInt32Type>(nctaidx, srnctaidx));

		auto lastGlobalThreadIndex = resources->template AllocateTemporary<PTX::UInt32Type>();

		auto multiply = new PTX::MultiplyInstruction<PTX::UInt32Type>(lastGlobalThreadIndex, ntidx, nctaidx);
		multiply->SetLower(true);
		this->m_builder.AddStatement(multiply);
		this->m_builder.AddStatement(new PTX::SubtractInstruction<PTX::UInt32Type>(lastGlobalThreadIndex, lastGlobalThreadIndex, new PTX::UInt32Value(1)));

		auto outputSizeLabel = this->m_builder.CreateLabel("RET_SIZE");
		auto outputSizePredicate = resources->template AllocateTemporary<PTX::PredicateType>();
		this->m_builder.AddStatement(new PTX::SetPredicateInstruction<PTX::UInt32Type>(
			outputSizePredicate, index, lastGlobalThreadIndex, PTX::UInt32Type::ComparisonOperator::NotEqual
		));
		this->m_builder.AddStatement(new PTX::BranchInstruction(outputSizeLabel, outputSizePredicate));

		// Compute the size (prefix sum + 1)

		auto outputSize = resources->template AllocateTemporary<PTX::UInt32Type>();
		this->m_builder.AddStatement(new PTX::AddInstruction<PTX::UInt32Type>(outputSize, changes, new PTX::UInt32Value(1)));

		// Get the size address and store

		UpdateSizeParameter(0, outputSize);
		UpdateSizeParameter(1, outputSize);

		this->m_builder.AddStatement(outputSizeLabel);
		this->m_builder.AddStatement(new PTX::BlankStatement());
	}

	void UpdateSizeParameter(unsigned int index, const PTX::Register<PTX::UInt32Type> *outputSize)
	{
		auto resources = this->m_builder.GetLocalResources();
		auto kernelResources = this->m_builder.GetKernelResources();

		auto parameter = kernelResources->GetParameter<PTX::PointerType<B, PTX::Int64Type>>(NameUtils::ReturnName(index));
		auto sizeParameter = kernelResources->GetParameter<PTX::PointerType<B, PTX::UInt32Type>>(NameUtils::SizeName(parameter));

		AddressGenerator<B> addressGenerator(this->m_builder);
		auto sizeAddress = addressGenerator.template GenerateAddress<PTX::UInt32Type>(sizeParameter);

		this->m_builder.AddStatement(new PTX::StoreInstruction<B, PTX::UInt32Type, PTX::GlobalSpace>(sizeAddress, outputSize));
	}

	void GenerateWarpPrefixSum(const PTX::Register<PTX::UInt32Type> *laneIndex, const PTX::Register<PTX::UInt32Type> *value)
	{
		auto resources = this->m_builder.GetLocalResources();
		auto& targetOptions = this->m_builder.GetTargetOptions();
		
		// Iteratively construct the warp prefix sum, shuffling values. This generates the unrolled loop

		for (auto offset = 1; offset < targetOptions.WarpSize; offset <<= 1)
		{
			// Shuffle the value from the above lane, merging if it part of the prefix sum

			auto temp = resources->template AllocateTemporary<PTX::UInt32Type>();
			this->m_builder.AddStatement(new PTX::ShuffleInstruction<PTX::UInt32Type>(
				temp, value, new PTX::UInt32Value(offset), new PTX::UInt32Value(0), -1, PTX::ShuffleInstruction<PTX::UInt32Type>::Mode::Up
			));

			// Check if part of prefix sum

			auto addPredicate = resources->template AllocateTemporary<PTX::PredicateType>();
			this->m_builder.AddStatement(new PTX::SetPredicateInstruction<PTX::UInt32Type>(
				addPredicate, laneIndex, new PTX::UInt32Value(offset), PTX::UInt32Type::ComparisonOperator::GreaterEqual
			));

			// Merge!

			auto add = new PTX::AddInstruction<PTX::UInt32Type>(value, temp, value);
			add->SetPredicate(addPredicate);
			this->m_builder.AddStatement(add);
		}
	}
};

}
