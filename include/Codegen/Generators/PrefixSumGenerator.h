#pragma once

#include "Codegen/Generators/Generator.h"

#include "Codegen/Generators/AddressGenerator.h"
#include "Codegen/Generators/IndexGenerator.h"

#include "HorseIR/Tree/Tree.h"

#include "PTX/PTX.h"

namespace Codegen {

template<PTX::Bits B>
class PrefixSumGenerator : public Generator
{
public:
	using Generator::Generator;

	template<class T>
	const PTX::Register<T> *Generate(const PTX::Address<B, T, PTX::GlobalSpace> *g_prefixSumAddress, const PTX::Register<T> *value, const PTX::TypedOperand<PTX::UInt32Type> *blockIndex = nullptr)
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
		// we have no guarantee of thread-block ordering, either use a global variable to allocate
		// blocks on an arrival basis to guarantee ordering - or have an unordered output

		auto resources = this->m_builder.GetLocalResources();
		auto kernelResources = this->m_builder.GetKernelResources();
		auto globalResources = this->m_builder.GetGlobalResources();

		auto& targetOptions = this->m_builder.GetTargetOptions();

		AddressGenerator<B> addressGenerator(this->m_builder);
		IndexGenerator indexGenerator(this->m_builder);
		auto index = indexGenerator.GenerateGlobalIndex(blockIndex);

		// Completed determining size

		auto prefixSum = resources->template AllocateTemporary<T>();
		this->m_builder.AddStatement(new PTX::MoveInstruction<T>(prefixSum, value));

		// Iteratively compute the prefix sum for the warp

		auto laneIndex = indexGenerator.GenerateLaneIndex();
		GenerateWarpPrefixSum(laneIndex, prefixSum);

		// For each warp, store the last (laneIndex == 31) value in shared memory at its warp index. Adapt from array to variable for addressing

		auto s_prefixSum = new PTX::ArrayVariableAdapter<T, 32, PTX::SharedSpace>(
			kernelResources->template AllocateSharedVariable<PTX::ArrayType<T, 32>>(this->m_builder.UniqueIdentifier("warpPrefixSums"))
		);

		auto warpIndex = indexGenerator.GenerateWarpIndex();
		auto s_prefixSumWarpAddress = addressGenerator.GenerateAddress(s_prefixSum, warpIndex);

		auto warpStoreLabel = this->m_builder.CreateLabel("WARP_STORE");
		auto lastLanePredicate = resources->template AllocateTemporary<PTX::PredicateType>();

		this->m_builder.AddStatement(new PTX::SetPredicateInstruction<PTX::UInt32Type>(
			lastLanePredicate, laneIndex, new PTX::UInt32Value(targetOptions.WarpSize - 1), PTX::UInt32Type::ComparisonOperator::NotEqual
		));
		this->m_builder.AddStatement(new PTX::BranchInstruction(warpStoreLabel, lastLanePredicate));
		this->m_builder.AddStatement(new PTX::StoreInstruction<B, T, PTX::SharedSpace>(s_prefixSumWarpAddress, prefixSum));

		// Synchronize the result so all values are visible to the first warp

		this->m_builder.AddStatement(new PTX::BlankStatement());
		this->m_builder.AddStatement(warpStoreLabel);
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

		auto warpLocalPrefixSum = resources->template AllocateTemporary<T>();
		this->m_builder.AddStatement(new PTX::LoadInstruction<B, T, PTX::SharedSpace>(warpLocalPrefixSum, s_prefixSumLaneAddress));

		// Prefix sum on the first warp

		GenerateWarpPrefixSum(laneIndex, warpLocalPrefixSum);

		// Store the value back and synchronize between all warps

		this->m_builder.AddStatement(new PTX::StoreInstruction<B, T, PTX::SharedSpace>(s_prefixSumLaneAddress, warpLocalPrefixSum));

		this->m_builder.AddStatement(new PTX::BlankStatement());
		this->m_builder.AddStatement(blockSumLabel);
		this->m_builder.AddStatement(new PTX::BarrierInstruction(new PTX::UInt32Value(0)));

		// If we are not the first warp, add the summed result from the shared memory - this completes the prefix sum for the block (!predicate)

		auto warpRestoreLabel = this->m_builder.CreateLabel("WARP_RESTORE");
		this->m_builder.AddStatement(new PTX::BranchInstruction(warpRestoreLabel, firstWarpPredicate, true));

		// Get the prefix sum (inclusive) from the previous warp

		auto warpPrefixSum = resources->template AllocateTemporary<T>();
		auto s_prefixSumWarpAddressM1 = s_prefixSumWarpAddress->CreateOffsetAddress(-1);
		this->m_builder.AddStatement(new PTX::LoadInstruction<B, T, PTX::SharedSpace>(warpPrefixSum, s_prefixSumWarpAddressM1));

		// Add to each value within the warp - computing the block local prefix sum

		this->m_builder.AddStatement(new PTX::AddInstruction<T>(prefixSum, prefixSum, warpPrefixSum));
		this->m_builder.AddStatement(new PTX::BlankStatement());
		this->m_builder.AddStatement(warpRestoreLabel);

		// For each block, load the previous block's value once it is completed. This forms a linear chain, but is fairly efficient

		auto blockPrefixSum = resources->template AllocateTemporary<T>();
		auto s_blockPrefixSum = kernelResources->template AllocateSharedVariable<T>(this->m_builder.UniqueIdentifier("blockPefixSum"));
		auto s_blockPrefixSumAddress = addressGenerator.GenerateAddress(s_blockPrefixSum);

		// To join loading/storing into a single section, the LAST thread of a block is responsible for loading/storing

		auto ntidx = resources->template AllocateTemporary<PTX::UInt32Type>();
		auto srntidx = new PTX::IndexedRegister4<PTX::UInt32Type>(PTX::SpecialRegisterDeclaration_ntid->GetVariable("%ntid"), PTX::VectorElement::X);
		this->m_builder.AddStatement(new PTX::MoveInstruction<PTX::UInt32Type>(ntidx, srntidx));

		auto localIndex = indexGenerator.GenerateLocalIndex();
		auto lastLocalIndex = resources->template AllocateTemporary<PTX::UInt32Type>();
		this->m_builder.AddStatement(new PTX::SubtractInstruction<PTX::UInt32Type>(lastLocalIndex, ntidx, new PTX::UInt32Value(1)));

		auto propagateLabel = this->m_builder.CreateLabel("PROPAGATE");
		auto propagatePredicate = resources->template AllocateTemporary<PTX::PredicateType>();
		this->m_builder.AddStatement(new PTX::SetPredicateInstruction<PTX::UInt32Type>(
			propagatePredicate, localIndex, lastLocalIndex, PTX::UInt32Type::ComparisonOperator::NotEqual
		));
		this->m_builder.AddStatement(new PTX::BranchInstruction(propagateLabel, propagatePredicate));

		if (blockIndex == nullptr)
		{
			this->m_builder.AddStatement(new PTX::AtomicInstruction<B, T, PTX::GlobalSpace, T::AtomicOperation::Add>(
				blockPrefixSum, g_prefixSumAddress, prefixSum
			));

			this->m_builder.AddStatement(new PTX::StoreInstruction<B, T, PTX::SharedSpace>(s_blockPrefixSumAddress, blockPrefixSum));
		}
		else
		{
			// Atomically check if the previous block has completed by checking a global counter

			auto atomicStartLabel = this->m_builder.CreateLabel("START");
			this->m_builder.AddStatement(new PTX::BlankStatement());
			this->m_builder.AddStatement(atomicStartLabel);

			auto completedBlocks = resources->template AllocateTemporary<PTX::UInt32Type>();
			auto g_completedBlocks = globalResources->template AllocateGlobalVariable<PTX::UInt32Type>(this->m_builder.UniqueIdentifier("completedBlocks"));
			auto g_completedBlocksAddress = addressGenerator.GenerateAddress(g_completedBlocks);

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

			this->m_builder.AddStatement(new PTX::LoadInstruction<B, T, PTX::GlobalSpace>(blockPrefixSum, g_prefixSumAddress));
			this->m_builder.AddStatement(new PTX::StoreInstruction<B, T, PTX::SharedSpace>(s_blockPrefixSumAddress, blockPrefixSum));

			// Update the global prefix sum to include this block

			this->m_builder.AddStatement(new PTX::ReductionInstruction<B, T, PTX::GlobalSpace, T::AtomicOperation::Add>(
				g_prefixSumAddress, prefixSum
			));

			// Proceed to next thread by incrementing the global counter

			this->m_builder.AddStatement(new PTX::AddInstruction<PTX::UInt32Type>(completedBlocks, completedBlocks, new PTX::UInt32Value(1)));
			this->m_builder.AddStatement(new PTX::StoreInstruction<B, PTX::UInt32Type, PTX::GlobalSpace>(g_completedBlocksAddress, completedBlocks));
		}

		this->m_builder.AddStatement(new PTX::BlankStatement());
		this->m_builder.AddStatement(propagateLabel);

		// Synchronize the results - every thread now has the previous thread's (inclusive) prefix sum

		this->m_builder.AddStatement(new PTX::BarrierInstruction(new PTX::UInt32Value(0)));

		// Load the prefix sum from the shared memory, and add to the current prefix sum. This is the GLOBAL prefix sum!

		this->m_builder.AddStatement(new PTX::LoadInstruction<B, T, PTX::SharedSpace>(blockPrefixSum, s_blockPrefixSumAddress));
		this->m_builder.AddStatement(new PTX::AddInstruction<T>(prefixSum, blockPrefixSum, prefixSum));

		return prefixSum;
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
