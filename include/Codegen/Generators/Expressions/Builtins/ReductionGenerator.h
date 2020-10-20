#pragma once

#include <limits>

#include "Codegen/Generators/Expressions/Builtins/BuiltinGenerator.h"

#include "Codegen/Builder.h"
#include "Codegen/Generators/Expressions/ConversionGenerator.h"
#include "Codegen/Generators/Expressions/OperandGenerator.h"
#include "Codegen/Generators/Expressions/ShuffleGenerator.h"
#include "Codegen/Generators/Indexing/AddressGenerator.h"
#include "Codegen/Generators/Indexing/DataIndexGenerator.h"
#include "Codegen/Generators/Indexing/ThreadGeometryGenerator.h"
#include "Codegen/Generators/Indexing/ThreadIndexGenerator.h"
#include "Codegen/Generators/Synchronization/BarrierGenerator.h"

#include "HorseIR/Analysis/Shape/Shape.h"
#include "HorseIR/Analysis/Shape/ShapeUtils.h"

#include "HorseIR/Tree/Tree.h"

#include "PTX/PTX.h"

#include "Utils/Math.h"

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

	std::string Name() const override { return "ReductionGenerator"; }

	// The output of a reduction function has no compression predicate. We therefore do not implement GenerateCompressionPredicate in this subclass

	const PTX::Register<T> *Generate(const HorseIR::LValue *target, const std::vector<HorseIR::Operand *>& arguments) override
	{
		auto targetRegister = this->GenerateTargetRegister(target, arguments);
		if constexpr(std::is_same<T, PTX::PredicateType>::value || std::is_same<T, PTX::Int8Type>::value)
		{
			// 8-bit integer and boolean values are only used directly in min/max. We can therefore safely
			// convert to 16-bit values without any changes in result

			if (m_reductionOp == ReductionOperation::Average || m_reductionOp == ReductionOperation::Average)
			{
				BuiltinGenerator<B, T>::Unimplemented("reduction operation " + ReductionOperationString(m_reductionOp));
			}

			auto resources = this->m_builder.GetLocalResources();
			auto targetRegister16 = resources->template AllocateTemporary<PTX::Int16Type>();

			ReductionGenerator<B, PTX::Int16Type> generator(this->m_builder, m_reductionOp);
			generator.Generate(targetRegister16, arguments);
			ConversionGenerator::ConvertSource<T, PTX::Int16Type>(this->m_builder, targetRegister, targetRegister16);
		}
		else
		{
			Generate(targetRegister, arguments);
		}
		return targetRegister;
	}

	void Generate(const PTX::Register<T> *targetRegister, const std::vector<HorseIR::Operand *>& arguments)
	{
		auto resources = this->m_builder.GetLocalResources();
		auto& inputOptions = this->m_builder.GetInputOptions();

		// Get the initial value for reduction

		OperandGenerator<B, T> opGenerator(this->m_builder);

		// Check if there is a compression predicate on the input value. If so, mask out the initial load according to the predicate

		OperandCompressionGenerator compGenerator(this->m_builder);
		auto compress = compGenerator.GetCompressionRegister(arguments.at(0));

		const PTX::TypedOperand<T> *src = nullptr;
		if (m_reductionOp == ReductionOperation::Length)
		{
			// No compression, we can use the active data size

			if (compress == nullptr)
			{
				ThreadGeometryGenerator<B> geometryGenerator(this->m_builder);
				auto dataSize = geometryGenerator.GenerateDataGeometry();

				auto convertedSize = ConversionGenerator::ConvertSource<T, PTX::UInt32Type>(this->m_builder, dataSize);
				this->m_builder.AddStatement(new PTX::MoveInstruction<T>(targetRegister, convertedSize));

				resources->SetReductionRegister(targetRegister, RegisterReductionGranularity::Single, RegisterReductionOperation::None);
				return;
			}

			// A count reduction is value agnostic performs a sum over 1's, one value for each active thread

			src = new PTX::Value<T>(1);
		}
		else
		{
			// All other reductions use the value for the thread

			src = opGenerator.GenerateOperand(arguments.at(0), OperandGenerator<B, T>::LoadKind::Vector);
		}

		// Load the underlying data size, ensuring it is within bounds

		auto inputPredicate = resources->template AllocateTemporary<PTX::PredicateType>();

		ThreadGeometryGenerator<B> geometryGenerator(this->m_builder);
		auto dataSize = geometryGenerator.GenerateDataGeometry();

		DataIndexGenerator<B> indexGenerator(this->m_builder);
		auto dataIndex = indexGenerator.GenerateDataIndex();

		this->m_builder.AddStatement(new PTX::SetPredicateInstruction<PTX::UInt32Type>(inputPredicate, dataIndex, dataSize, PTX::UInt32Type::ComparisonOperator::Less));

		// Select the initial value for the reduction depending on the data size and compression mask

		if (compress == nullptr)
		{
			// If no compression is active, select between the source and the null value depending on the compression

			this->m_builder.AddStatement(new PTX::SelectInstruction<T>(targetRegister, src, GenerateNullValue(m_reductionOp), inputPredicate));
		}
		else
		{
			// If the input is within range, select between the source and null values depending on the compression

			auto s1 = new PTX::SelectInstruction<T>(targetRegister, src, GenerateNullValue(m_reductionOp), compress);
			s1->SetPredicate(inputPredicate);
			this->m_builder.AddStatement(s1);

			// If the input is out of range, set the index to the null value

			auto s2 = new PTX::MoveInstruction<T>(targetRegister, GenerateNullValue(m_reductionOp));
			s2->SetPredicate(inputPredicate, true);
			this->m_builder.AddStatement(s2);
		}

		auto& codeOptions = this->m_builder.GetCodegenOptions();
		switch (codeOptions.Reduction)
		{
			case CodegenOptions::ReductionKind::ShuffleBlock:
			{
				GenerateShuffleBlock(targetRegister);
				break;
			}
			case CodegenOptions::ReductionKind::ShuffleWarp:
			{
				GenerateShuffleWarp(targetRegister);
				break;
			}
			case CodegenOptions::ReductionKind::Shared:
			{
				GenerateShared(targetRegister);
				break;
			}
		}
	}

	void GenerateShuffleReduction(const PTX::Register<T> *target, std::int32_t activeWarps)
	{
		// Warp shuffle the values down, reducing at each level until a single value is computed, log_2(WARP_SZ)

		auto warpSize = this->m_builder.GetTargetOptions().WarpSize;

		auto& kernelOptions = this->m_builder.GetKernelOptions();
		kernelOptions.SetThreadMultiple(activeWarps);

		for (unsigned int offset = activeWarps >> 1; offset > 0; offset >>= 1)
		{
			// Generate the shuffle instruction to pull down the other value

			ShuffleGenerator<T> shuffleGenerator(this->m_builder);
			auto temp = shuffleGenerator.Generate(target, offset, warpSize - 1, -1, PTX::ShuffleInstruction::Mode::Down);

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

	void GenerateShuffleBlock(const PTX::Register<T> *target)
	{
		auto resources = this->m_builder.GetLocalResources();
		auto globalResources = this->m_builder.GetGlobalResources();

		// Allocate shared memory with 1 slot per warp

		auto& kernelOptions = this->m_builder.GetKernelOptions();
		auto& targetOptions = this->m_builder.GetTargetOptions();
		auto& inputOptions = this->m_builder.GetInputOptions();

		int warpSize = targetOptions.WarpSize;

		auto blockSize = Utils::Math::RoundUp(GetBlockSize(), warpSize);
		kernelOptions.SetBlockSize(blockSize);

		auto sharedMemorySize = blockSize / warpSize;
		auto sharedMemory = globalResources->template AllocateDynamicSharedMemory<T>(sharedMemorySize);

		// Generate the shuffle for each warp

		GenerateShuffleReduction(target, warpSize);

		// Write a single reduced value to shared memory for each warp

		ThreadIndexGenerator<B> indexGenerator(this->m_builder);
		auto laneIndex = indexGenerator.GenerateLaneIndex();

		// Check if we are the first lane in the warp

		auto predWarp = resources->template AllocateTemporary<PTX::PredicateType>();
		auto labelWarp = this->m_builder.CreateLabel("RED_WARP");

		this->m_builder.AddStatement(new PTX::SetPredicateInstruction<PTX::UInt32Type>(predWarp, laneIndex, new PTX::UInt32Value(0), PTX::UInt32Type::ComparisonOperator::NotEqual));
		this->m_builder.AddStatement(new PTX::BranchInstruction(labelWarp, predWarp));
		this->m_builder.AddStatement(new PTX::BlankStatement());

		// Store the value in shared memory

		auto warpIndex = indexGenerator.GenerateWarpIndex();

		AddressGenerator<B, T> addressGenerator(this->m_builder);
		auto sharedWarpAddress = addressGenerator.GenerateAddress(sharedMemory, warpIndex);

		this->m_builder.AddStatement(new PTX::StoreInstruction<B, T, PTX::SharedSpace, PTX::StoreSynchronization::Volatile>(sharedWarpAddress, target));

		// End the if statement

		this->m_builder.AddStatement(new PTX::BlankStatement());
		this->m_builder.AddStatement(labelWarp);

		// Synchronize all values in shared memory from across warps

		BarrierGenerator<B> barrierGenerator(this->m_builder);
		barrierGenerator.Generate();

		// Check if we are the first warp in the block for the final part of the reduction

		auto warpid = indexGenerator.GenerateWarpIndex();
		auto cellWarps = blockSize / warpSize;

		auto cellWarp = resources->template AllocateTemporary<PTX::UInt32Type>();
		this->m_builder.AddStatement(new PTX::RemainderInstruction<PTX::UInt32Type>(cellWarp, warpid, new PTX::UInt32Value(cellWarps)));

		auto predBlock = resources->template AllocateTemporary<PTX::PredicateType>();
		auto labelBlock = this->m_builder.CreateLabel("RED_BLOCK");

		this->m_builder.AddStatement(new PTX::SetPredicateInstruction<PTX::UInt32Type>(predBlock, cellWarp, new PTX::UInt32Value(0), PTX::UInt32Type::ComparisonOperator::NotEqual));
		this->m_builder.AddStatement(new PTX::BranchInstruction(labelBlock, predBlock));
		this->m_builder.AddStatement(new PTX::BlankStatement());

		// Load the values back from the shared memory into the individual threads

		auto sharedLaneAddress = addressGenerator.template GenerateAddress<PTX::SharedSpace>(sharedMemory, laneIndex);
		this->m_builder.AddStatement(new PTX::LoadInstruction<B, T, PTX::SharedSpace, PTX::LoadSynchronization::Volatile>(target, sharedLaneAddress));

		ThreadGeometryGenerator<B> geometryGenerator(this->m_builder);
		auto warpCount = geometryGenerator.GenerateWarpCount();

		auto predActive = resources->template AllocateTemporary<PTX::PredicateType>();
		this->m_builder.AddStatement(new PTX::SetPredicateInstruction<PTX::UInt32Type>(predActive, laneIndex, warpCount, PTX::UInt32Type::ComparisonOperator::Less));
		this->m_builder.AddStatement(new PTX::SelectInstruction<T>(target, target, GenerateNullValue(m_reductionOp), predActive));

		// Reduce the individual values from all warps into 1 final value for the block.
		// To handle multiple cells per block, only reduce in segments (# warps per cell)

		GenerateShuffleReduction(target, cellWarps);

		if (inputOptions.IsListGeometry())
		{
			this->m_builder.AddStatement(new PTX::StoreInstruction<B, T, PTX::SharedSpace, PTX::StoreSynchronization::Volatile>(sharedLaneAddress, target));
		}

		// End the if statement

		this->m_builder.AddStatement(new PTX::BlankStatement());
		this->m_builder.AddStatement(labelBlock);

		if (inputOptions.IsListGeometry())
		{
			barrierGenerator.Generate();

			auto reload = new PTX::LoadInstruction<B, T, PTX::SharedSpace, PTX::LoadSynchronization::Volatile>(target, sharedWarpAddress);
			reload->SetPredicate(predWarp, true);
			this->m_builder.AddStatement(reload);
		}

		// Write a single value per block to the global atomic value

		resources->SetReductionRegister(target, RegisterReductionGranularity::Block, GetRegisterReductionOperation(m_reductionOp));
	}

	void GenerateShuffleWarp(const PTX::Register<T> *target)
	{
		// Generate the shuffle for each warp

		auto warpSize = this->m_builder.GetTargetOptions().WarpSize;
		GenerateShuffleReduction(target, warpSize);

		// Write a single value per warp to the global atomic value

		auto resources = this->m_builder.GetLocalResources();
		resources->SetReductionRegister(target, RegisterReductionGranularity::Warp, GetRegisterReductionOperation(m_reductionOp));
	}

	void GenerateShared(const PTX::Register<T> *target)
	{
		// Fetch generation state and options

		auto resources = this->m_builder.GetLocalResources();
		auto globalResources = this->m_builder.GetGlobalResources();

		auto& targetOptions = this->m_builder.GetTargetOptions();
		auto& inputOptions = this->m_builder.GetInputOptions();
		auto& kernelOptions = this->m_builder.GetKernelOptions();

		// Compute the number of threads used in this reduction

		auto blockSize = Utils::Math::Power2(GetBlockSize());
		kernelOptions.SetBlockSize(blockSize);

		auto sharedMemory = globalResources->template AllocateDynamicSharedMemory<T>(blockSize);

		// Load the the local thread index for accessing the shared memory

		ThreadIndexGenerator<B> indexGenerator(this->m_builder);
		auto localIndex = indexGenerator.GenerateLocalIndex();
		auto activeIndex = (inputOptions.IsVectorGeometry()) ? indexGenerator.GenerateLocalIndex() : indexGenerator.GenerateListLocalIndex();

		AddressGenerator<B, T> addressGenerator(this->m_builder);
		auto sharedThreadAddress = addressGenerator.GenerateAddress(sharedMemory, localIndex);

		// Load the initial value into shared memory

		this->m_builder.AddStatement(new PTX::StoreInstruction<B, T, PTX::SharedSpace, PTX::StoreSynchronization::Volatile>(sharedThreadAddress, target));

		// Synchronize the thread group for a consistent view of shared memory

		BarrierGenerator<B> barrierGenerator(this->m_builder);
		barrierGenerator.Generate();

		// Perform an iterative reduction on the shared memory in a pyramid type fashion

		int warpSize = targetOptions.WarpSize;

		for (unsigned int i = (blockSize >> 1); i >= warpSize; i >>= 1)
		{
			// At each level, half the threads become inactive

			auto pred = resources->template AllocateTemporary<PTX::PredicateType>();
			auto guard = new PTX::UInt32Value(i - 1);

			auto name = (i <= warpSize) ? "RED_STORE" : "RED_" + std::to_string(i);
			auto label = this->m_builder.CreateLabel(name);

			// Check to see if we are an active thread

			this->m_builder.AddStatement(new PTX::SetPredicateInstruction<PTX::UInt32Type>(pred, activeIndex, guard, PTX::UInt32Type::ComparisonOperator::Higher));
			this->m_builder.AddStatement(new PTX::BranchInstruction(label, pred));
			this->m_builder.AddStatement(new PTX::BlankStatement());

			if (i <= warpSize)
			{
				// Once the active thread count fits within a warp, reduce without synchronization

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

				barrierGenerator.Generate();
			}
		}

		// Wrie a single value per block to the global atomic variable

		resources->SetReductionRegister(target, RegisterReductionGranularity::Block, GetRegisterReductionOperation(m_reductionOp));
	}

private:

	std::uint32_t GetBlockSize() const
	{
		auto& inputOptions = this->m_builder.GetInputOptions();
		auto& targetOptions = this->m_builder.GetTargetOptions();

		// Determine the block size, based on the machine characteristics and data size

		auto blockSize = targetOptions.MaxBlockSize;
		if (const auto vectorGeometry = HorseIR::Analysis::ShapeUtils::GetShape<HorseIR::Analysis::VectorShape>(inputOptions.ThreadGeometry))
		{
			if (const auto constantSize = HorseIR::Analysis::ShapeUtils::GetSize<HorseIR::Analysis::Shape::ConstantSize>(vectorGeometry->GetSize()))
			{
				// Allocate a smaller number of threads if possible, otherwise use the entire block

				auto size = constantSize->GetValue();
				if (size < blockSize)
				{
					blockSize = size;
				}
			}
		}
		else if (HorseIR::Analysis::ShapeUtils::IsShape<HorseIR::Analysis::ListShape>(inputOptions.ThreadGeometry))
		{
			// For each cell, limit by the number of cell threads if specified, otherwise us the entire block for each cell

			auto size = inputOptions.ListCellThreads;
			if (size != InputOptions::DynamicSize && size < blockSize)
			{
				blockSize = size;
			}
		}
		else
		{
			BuiltinGenerator<B, T>::Unimplemented("reduction block size for thread geometry " + HorseIR::Analysis::ShapeUtils::ShapeString(inputOptions.ThreadGeometry));
		}
		return blockSize;
	}

	RegisterReductionOperation GetRegisterReductionOperation(ReductionOperation reductionOp) const
	{
		switch (reductionOp)
		{
			case ReductionOperation::Length:
			case ReductionOperation::Average:
			case ReductionOperation::Sum:
				return RegisterReductionOperation::Add;
			case ReductionOperation::Minimum:
				return RegisterReductionOperation::Minimum;
			case ReductionOperation::Maximum:
				return RegisterReductionOperation::Maximum;
			default:
				BuiltinGenerator<B, T>::Unimplemented("reduction operation " + ReductionOperationString(reductionOp));
		}
	}
	
	const PTX::Value<T> *GenerateNullValue(ReductionOperation reductionOp) const
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
		auto offsetAddress = address->CreateOffsetAddress(offset);

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
