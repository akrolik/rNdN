#pragma once

#include <cmath>
#include <limits>

#include "Codegen/Generators/Expressions/Builtins/BuiltinGenerator.h"

#include "Codegen/Builder.h"
#include "Codegen/Generators/AddressGenerator.h"
#include "Codegen/Generators/IndexGenerator.h"
#include "Codegen/Generators/SizeGenerator.h"
#include "Codegen/Generators/Expressions/ConversionGenerator.h"
#include "Codegen/Generators/Expressions/OperandGenerator.h"

#include "PTX/PTX.h"

namespace Codegen {

enum class ReductionOperation {
	Length,
	Sum,
	Average,
	Minimum,
	Maximum
};

static std::string ReductionOperationString(ReductionOperation reductionOp)
{
	switch (reductionOp)
	{
		case ReductionOperation::Length:
			return "length";
		case ReductionOperation::Sum:
			return "sum";
		case ReductionOperation::Average:
			return "avg";
		case ReductionOperation::Minimum:
			return "min";
		case ReductionOperation::Maximum:
			return "max";
	}
	return "<unknown>";
}

template<PTX::Bits B, class T>
class ReductionGenerator : public BuiltinGenerator<B, T>
{
public:
	ReductionGenerator(Builder& builder, ReductionOperation reductionOp) : BuiltinGenerator<B, T>(builder), m_reductionOp(reductionOp) {}

	// The output of a reduction function has no compression predicate. We therefore do not implement
	// GenerateCompressionPredicate in this subclass


	void Generate(const PTX::Register<T> *target, const HorseIR::CallExpression *call) override
	{
		//TODO: Support correct type matrix for reductions
		if constexpr(!std::is_same<T, PTX::PredicateType>::value && T::TypeBits != PTX::Bits::Bits8 && T::TypeBits != PTX::Bits::Bits16)
		{
		auto resources = this->m_builder.GetLocalResources();

		// Load the underlying data size

		SizeGenerator<B> sizeGen(this->m_builder);
		auto size = sizeGen.GenerateInputSize();

		// Load the the global thread index for checking the data bounds

		IndexGenerator indexGen(this->m_builder);
		auto globalIndex = indexGen.GenerateGlobalIndex();

		// Some operations require a 64-bit value for comparison

		auto globalIndex64 = resources->template AllocateTemporary<PTX::UInt64Type>();
		this->m_builder.AddStatement(new PTX::ConvertInstruction<PTX::UInt64Type, PTX::UInt32Type>(globalIndex64, globalIndex));

		// Check if the thread is in bounds for the input data

		auto inputPredicate = resources->template AllocateTemporary<PTX::PredicateType>();
		this->m_builder.AddStatement(new PTX::SetPredicateInstruction<PTX::UInt64Type>(inputPredicate, globalIndex64, size, PTX::UInt64Type::ComparisonOperator::Less));

		// Get the initial value for reduction and store it in the shared memory

		OperandGenerator<B, T> opGen(this->m_builder);

		// Check if there is a compression predicate on the input value. If so, mask out the
		// initial load according to the predicate

		OperandCompressionGenerator compGen(this->m_builder);
		auto compress = compGen.GetCompressionRegister(call->GetArgument(0));

		const PTX::TypedOperand<T> *src = nullptr;
		if (m_reductionOp == ReductionOperation::Length)
		{
			// A count reduction is value agnostic performs a sum over 1's, one value for each active thread

			src = new PTX::Value<T>(1);
		}
		else
		{
			// All other reductions use the value for the thread

			src = opGen.GenerateOperand(call->GetArgument(0));
		}

		// Select the initial value for the reduction depending on the data size and compression mask

		if (compress == nullptr)
		{
			// If no compression is active, select between the source and the null value depending on the compression

			this->m_builder.AddStatement(new PTX::SelectInstruction<T>(target, src, GenerateNullValue(m_reductionOp), inputPredicate));
		}
		else
		{
			// If the input is within range, select between the source and null values depending on the compression

			auto s1 = new PTX::SelectInstruction<T>(target, src, GenerateNullValue(m_reductionOp), compress);
			s1->SetPredicate(inputPredicate);
			this->m_builder.AddStatement(s1);

			// If the input is out of range, set the index to the null value

			auto s2 = new PTX::MoveInstruction<T>(target, GenerateNullValue(m_reductionOp));
			s2->SetPredicate(inputPredicate, true);
			this->m_builder.AddStatement(s2);
		}

		//TODO: Select which type of reduction we want to implement
		GenerateShuffleBlock(target);
		// GenerateShuffleWarp(target);
		// GenerateShared(target);
		}
	}

	void GenerateShuffleReduction(const PTX::Register<T> *target)
	{
		// Warp shuffle the values down, reducing at each level until a single value is computed, log_2(WARP_SZ)

		auto resources = this->m_builder.GetLocalResources();
		auto warpSize = this->m_builder.GetTargetOptions().WarpSize;

		auto& kernelOptions = this->m_builder.GetKernelOptions();
		kernelOptions.SetThreadMultiple(warpSize);

		for (unsigned int offset = warpSize >> 1; offset > 0; offset >>= 1)
		{
			// Generate the shuffle instruction to pull down the other value

			auto temp = GenerateShuffle(target, offset, warpSize);

			// Generate the operation for the reduction

			auto [result, predicate] = GenerateReductionInstruction(target, temp);

			// If a predicate is present, then we are responsible for either moving the value into place or storing

			if (predicate != nullptr)
			{
				auto move = new PTX::MoveInstruction<T>(target, result); 
				move->SetPredicate(predicate);
				this->m_builder.AddStatement(move);
			}
		}
	}

	const PTX::Register<T> *GenerateShuffle(const PTX::Register<T> *value, unsigned int offset, unsigned int warpSize)
	{
		auto resources = this->m_builder.GetLocalResources();
		auto temp = resources->template AllocateTemporary<T>();

                if constexpr(T::TypeBits == PTX::Bits::Bits64)
		{
			// Shuffling only permits values of 32 bits. If we ave 64 bits, then split the value
			// into two sections, shuffle twice, and re-combine the shuffled result
			//
			// mov.b64 {%temp1,%temp2}, %in;
			// shfl.sync.down.b32 	%temp3, %temp1, ...;
			// shfl.sync.down.b32 	%temp4, %temp2, ...;
			// mov.b64 %out, {%temp3,%temp4};

			auto temp1 = resources->template AllocateTemporary<PTX::Bit32Type>();
			auto temp2 = resources->template AllocateTemporary<PTX::Bit32Type>();
			auto temp3 = resources->template AllocateTemporary<PTX::Bit32Type>();
			auto temp4 = resources->template AllocateTemporary<PTX::Bit32Type>();

			auto bracedInput = new PTX::Braced2Register<PTX::Bit32Type>({temp1, temp2});
			auto bracedOutput = new PTX::Braced2Operand<PTX::Bit32Type>({temp3, temp4});

			ConversionGenerator conversion(this->m_builder);
			auto valueBit = conversion.ConvertSource<PTX::Bit64Type, T>(value);
			auto tempBit = conversion.ConvertSource<PTX::Bit64Type, T>(temp);

			this->m_builder.AddStatement(new PTX::Unpack2Instruction<PTX::Bit64Type>(bracedInput, valueBit));
			this->m_builder.AddStatement(new PTX::ShuffleInstruction<PTX::Bit32Type>(
				temp3, temp1, new PTX::UInt32Value(offset), new PTX::UInt32Value(warpSize - 1), -1, PTX::ShuffleInstruction<PTX::Bit32Type>::Mode::Down
			));
			this->m_builder.AddStatement(new PTX::ShuffleInstruction<PTX::Bit32Type>(
				temp4, temp2, new PTX::UInt32Value(offset), new PTX::UInt32Value(warpSize - 1), -1, PTX::ShuffleInstruction<PTX::Bit32Type>::Mode::Down
			));
			this->m_builder.AddStatement(new PTX::Pack2Instruction<PTX::Bit64Type>(tempBit, bracedOutput));

		}
		else
		{
			this->m_builder.AddStatement(new PTX::ShuffleInstruction<T>(
				temp, value, new PTX::UInt32Value(offset), new PTX::UInt32Value(warpSize - 1), -1, PTX::ShuffleInstruction<T>::Mode::Down
			));
		}
		return temp;
	}

	void GenerateShuffleBlock(const PTX::Register<T> *target)
	{
		auto resources = this->m_builder.GetLocalResources();
		auto globalResources = this->m_builder.GetGlobalResources();

		// Allocate shared memory with 1 slot per warp

		auto& kernelOptions = this->m_builder.GetKernelOptions();
		auto& targetOptions = this->m_builder.GetTargetOptions();
		auto& inputOptions = this->m_builder.GetInputOptions();

		int warpSize = targetOptions.WarpSize;
		unsigned int blockSize = targetOptions.MaxBlockSize;
		if (inputOptions.ActiveThreads != InputOptions::DynamicSize && inputOptions.ActiveThreads < blockSize)
		{
			blockSize = ((inputOptions.ActiveThreads + warpSize - 1) / warpSize) * warpSize;
		}
		kernelOptions.SetBlockSize(blockSize);

		auto sharedMemorySize = blockSize / warpSize;
		auto sharedMemory = globalResources->template AllocateDynamicSharedMemory<T>(sharedMemorySize);

		// Generate the shuffle for each warp

		GenerateShuffleReduction(target);

		// Write a single reduced value to shared memory for each warp

		IndexGenerator indexGen(this->m_builder);
		auto laneid = indexGen.GenerateLaneIndex();

		// Check if we are the first lane in the warp

		auto predWarp = resources->template AllocateTemporary<PTX::PredicateType>();
		auto labelWarp = new PTX::Label("RED_WARP");
		auto branchWarp = new PTX::BranchInstruction(labelWarp);
		branchWarp->SetPredicate(predWarp);

		this->m_builder.AddStatement(new PTX::SetPredicateInstruction<PTX::UInt32Type>(predWarp, laneid, new PTX::UInt32Value(0), PTX::UInt32Type::ComparisonOperator::NotEqual));
		this->m_builder.AddStatement(branchWarp);
		this->m_builder.AddStatement(new PTX::BlankStatement());

		// Store the value in shared memory

		AddressGenerator<B> addressGenerator(this->m_builder);
		auto sharedWarpAddress = addressGenerator.template Generate<T, PTX::SharedSpace>(sharedMemory, AddressGenerator<B>::IndexKind::Warp);

		this->m_builder.AddStatement(new PTX::StoreInstruction<B, T, PTX::SharedSpace, PTX::StoreSynchronization::Volatile>(sharedWarpAddress, target));

		// End the if statement

		this->m_builder.AddStatement(new PTX::BlankStatement());
		this->m_builder.AddStatement(labelWarp);

		// Synchronize all values in shared memory from across warps

		this->m_builder.AddStatement(new PTX::BarrierInstruction(new PTX::UInt32Value(0), true));

		// Load the values back from the shared memory into the individual threads

		auto sharedLaneAddress = addressGenerator.template Generate<T, PTX::SharedSpace>(sharedMemory, AddressGenerator<B>::IndexKind::Lane);
		this->m_builder.AddStatement(new PTX::LoadInstruction<B, T, PTX::SharedSpace, PTX::LoadSynchronization::Volatile>(target, sharedLaneAddress));

		SizeGenerator<B> sizeGen(this->m_builder);
		auto warpCount = sizeGen.GenerateWarpCount();

		auto predActive = resources->template AllocateTemporary<PTX::PredicateType>();
		this->m_builder.AddStatement(new PTX::SetPredicateInstruction<PTX::UInt32Type>(predActive, laneid, warpCount, PTX::UInt32Type::ComparisonOperator::Less));
		this->m_builder.AddStatement(new PTX::SelectInstruction<T>(target, target, GenerateNullValue(m_reductionOp), predActive));

		// Check if we are the first warp in the block for the final part of the reduction

		auto warpid = indexGen.GenerateWarpIndex();

		auto predBlock = resources->template AllocateTemporary<PTX::PredicateType>();
		auto labelBlock = new PTX::Label("RED_BLOCK");
		auto branchBlock = new PTX::BranchInstruction(labelBlock);
		branchBlock->SetPredicate(predBlock);

		this->m_builder.AddStatement(new PTX::SetPredicateInstruction<PTX::UInt32Type>(predBlock, warpid, new PTX::UInt32Value(0), PTX::UInt32Type::ComparisonOperator::NotEqual));
		this->m_builder.AddStatement(branchBlock);
		this->m_builder.AddStatement(new PTX::BlankStatement());

		// Reduce the individual values from all warps into 1 final value for the block

		GenerateShuffleReduction(target);

		// End the if statement

		this->m_builder.AddStatement(new PTX::BlankStatement());
		this->m_builder.AddStatement(labelBlock);

		// Write a single value per block to the global atomic value

		resources->SetReductionRegister(target, RegisterAllocator::ReductionGranularity<T>::Block, GetRegisterReductionOperation(m_reductionOp));
	}

	void GenerateShuffleWarp(const PTX::Register<T> *target)
	{
		// Generate the shuffle for each warp

		GenerateShuffleReduction(target);

		// Write a single value per warp to the global atomic value

		auto resources = this->m_builder.GetLocalResources();
		resources->SetReductionRegister(target, RegisterAllocator::ReductionGranularity<T>::Warp, GetRegisterReductionOperation(m_reductionOp));
	}

	void GenerateShared(const PTX::Register<T> *target)
	{
		// Fetch generation state and options

		auto resources = this->m_builder.GetLocalResources();
		auto globalResources = this->m_builder.GetGlobalResources();

		auto& targetOptions = this->m_builder.GetTargetOptions();
		auto& inputOptions = this->m_builder.GetInputOptions();

		// Compute the number of threads used in this reduction

		auto& kernelOptions = this->m_builder.GetKernelOptions();

		unsigned int blockSize = targetOptions.MaxBlockSize;
		if (inputOptions.ActiveThreads != InputOptions::DynamicSize && inputOptions.ActiveThreads < blockSize)
		{
			blockSize = std::pow(2, std::ceil(std::log2(inputOptions.ActiveThreads)));
		}
		kernelOptions.SetBlockSize(blockSize);

		auto sharedMemory = globalResources->template AllocateDynamicSharedMemory<T>(blockSize);

		AddressGenerator<B> addressGenerator(this->m_builder);
		auto sharedThreadAddress = addressGenerator.template Generate<T, PTX::SharedSpace>(sharedMemory, AddressGenerator<B>::IndexKind::Local);

		// Load the the local thread index for accessing the shared memory

		IndexGenerator indexGen(this->m_builder);
		auto localIndex = indexGen.GenerateLocalIndex();

		// Load the initial value into shared memory

		this->m_builder.AddStatement(new PTX::StoreInstruction<B, T, PTX::SharedSpace, PTX::StoreSynchronization::Volatile>(sharedThreadAddress, target));

		// Synchronize the thread group for a consistent view of shared memory

		this->m_builder.AddStatement(new PTX::BarrierInstruction(new PTX::UInt32Value(0), true));

		// Perform an iterative reduction on the shared memory in a pyramid type fashion

		int warpSize = targetOptions.WarpSize;

		for (unsigned int i = (blockSize >> 1); i >= warpSize; i >>= 1)
		{
			// At each level, half the threads become inactive

			auto pred = resources->template AllocateTemporary<PTX::PredicateType>();
			auto guard = new PTX::UInt32Value(i - 1);
			auto label = new PTX::Label("RED_" + std::to_string(i));
			auto branch = new PTX::BranchInstruction(label);
			branch->SetPredicate(pred);

			// Check to see if we are an active thread

			this->m_builder.AddStatement(new PTX::SetPredicateInstruction<PTX::UInt32Type>(pred, localIndex, guard, PTX::UInt32Type::ComparisonOperator::Higher));
			this->m_builder.AddStatement(branch);
			this->m_builder.AddStatement(new PTX::BlankStatement());

			if (i <= warpSize)
			{
				// Once the active thread count fits within a warp, reduce without synchronization

				label->SetName("RED_STORE");
				for (unsigned int j = i; j >= 1; j >>= 1)
				{
					GenerateSharedReduction(target, sharedThreadAddress, j);
				}
			}
			else
			{
				GenerateSharedReduction(target, sharedThreadAddress, i);
			}

			this->m_builder.AddStatement(new PTX::BlankStatement());
			this->m_builder.AddStatement(label);
			if (i > warpSize)
			{
				// If we still have >1 warps running, synchronize the group since they may not be in lock-step

				this->m_builder.AddStatement(new PTX::BarrierInstruction(new PTX::UInt32Value(0), true));
			}
		}

		// Wrie a single value per block to the global atomic variable

		resources->SetReductionRegister(target, RegisterAllocator::ReductionGranularity<T>::Block, GetRegisterReductionOperation(m_reductionOp));
	}

private:
	static RegisterAllocator::ReductionOperation<T> GetRegisterReductionOperation(ReductionOperation reductionOp)
	{
		switch (reductionOp)
		{
			case ReductionOperation::Length:
			case ReductionOperation::Average:
			case ReductionOperation::Sum:
				return RegisterAllocator::ReductionOperation<T>::Add;
			case ReductionOperation::Minimum:
				return RegisterAllocator::ReductionOperation<T>::Minimum;
			case ReductionOperation::Maximum:
				return RegisterAllocator::ReductionOperation<T>::Maximum;
			default:
				BuiltinGenerator<B, T>::Unimplemented("reduction operation " + ReductionOperationString(reductionOp));
		}
	}
	
	static const PTX::Value<T> *GenerateNullValue(ReductionOperation reductionOp)
	{
		switch (reductionOp)
		{
			case ReductionOperation::Length:
			case ReductionOperation::Average:
			case ReductionOperation::Sum:
				return new PTX::Value<T>(0);
			case ReductionOperation::Minimum:
				return new PTX::Value<T>(std::numeric_limits<typename T::SystemType>::max());
			case ReductionOperation::Maximum:
				return new PTX::Value<T>(std::numeric_limits<typename T::SystemType>::min());
			default:
				BuiltinGenerator<B, T>::Unimplemented("reduction operation " + ReductionOperationString(reductionOp));
		}
	}

	void GenerateSharedReduction(const PTX::Register<T> *target, const PTX::Address<B, T, PTX::SharedSpace> *address, unsigned int offset)
	{
		// Load the 2 values for this stage of the reduction into registers using the provided offset. Volatile
		// reads/writes are used to ensure synchronization

		auto resources = this->m_builder.GetLocalResources();

		auto offsetVal = resources->template AllocateTemporary<T>();
		auto offsetAddress = address->CreateOffsetAddress(offset * PTX::BitSize<T::TypeBits>::NumBytes);

		this->m_builder.AddStatement(new PTX::LoadInstruction<B, T, PTX::SharedSpace, PTX::LoadSynchronization::Volatile>(target, address));
		this->m_builder.AddStatement(new PTX::LoadInstruction<B, T, PTX::SharedSpace, PTX::LoadSynchronization::Volatile>(offsetVal, offsetAddress));

		// Generate the reduction instruction

		auto [result, predicate] = GenerateReductionInstruction(target, offsetVal);

		if (offset > 1)
		{
			// Store the new value back to the shared memory

			auto store = new PTX::StoreInstruction<B, T, PTX::SharedSpace, PTX::StoreSynchronization::Volatile>(address, result); 
			store->SetPredicate(predicate);
			this->m_builder.AddStatement(store);
		}
		else
		{
			// As an optimization, if we are on the last step, do not store to shared memory - the value is in a register

			if (predicate != nullptr)
			{
				auto move = new PTX::MoveInstruction<T>(target, result); 
				move->SetPredicate(predicate);
				this->m_builder.AddStatement(move);
			}
		}
	}

	const std::pair<const PTX::Register<T>*, const PTX::Register<PTX::PredicateType> *> GenerateReductionInstruction(const PTX::Register<T> *target, const PTX::Register<T> *offsetVal)
	{
		auto resources = this->m_builder.GetLocalResources();

		// The predicate can (optionally) disable the subsequent store. This is used for comparison type reductions (min/max)

		switch (m_reductionOp)
		{
			case ReductionOperation::Length:
			case ReductionOperation::Average:
			case ReductionOperation::Sum:
			{
				// All 3 reductions are essentially a sum. Count on all 1's, average performs a final division

				this->m_builder.AddStatement(new PTX::AddInstruction<T>(target, target, offsetVal));
				return {target, nullptr};
			}
			case ReductionOperation::Minimum:
			{
				// Minimum reduction checks if the offset value is less than the initial value. If so, we
				// will write it back in place of the initial value

				auto predicate = resources->template AllocateTemporary<PTX::PredicateType>();
				this->m_builder.AddStatement(new PTX::SetPredicateInstruction<T>(predicate, offsetVal, target, T::ComparisonOperator::Less));
				return {offsetVal, predicate};
			}
			case ReductionOperation::Maximum:
			{
				// Maximum reduction checks if the offset value is greater than the initial value. If so, we
				// will write it back in place of the initial value

				auto predicate = resources->template AllocateTemporary<PTX::PredicateType>();
				this->m_builder.AddStatement(new PTX::SetPredicateInstruction<T>(predicate, offsetVal, target, T::ComparisonOperator::Greater));
				return {offsetVal, predicate};
			}
			default:
			{
				BuiltinGenerator<B, T>::Unimplemented("reduction operation " + ReductionOperationString(m_reductionOp));
			}
		}
	}

	ReductionOperation m_reductionOp;
};

}
