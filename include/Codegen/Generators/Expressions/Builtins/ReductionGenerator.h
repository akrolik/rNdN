#pragma once

#include <cmath>
#include <limits>

#include "Codegen/Generators/Expressions/Builtins/BuiltinGenerator.h"

#include "Codegen/Builder.h"
#include "Codegen/Generators/AddressGenerator.h"
#include "Codegen/Generators/IndexGenerator.h"
#include "Codegen/Generators/ParameterGenerator.h"
#include "Codegen/Generators/ReturnGenerator.h"
#include "Codegen/Generators/SizeGenerator.h"
#include "Codegen/Generators/Expressions/OperandGenerator.h"

#include "PTX/StateSpace.h"
#include "PTX/Type.h"
#include "PTX/Declarations/Declaration.h"
#include "PTX/Declarations/SpecialRegisterDeclarations.h"
#include "PTX/Instructions/DevInstruction.h"
#include "PTX/Instructions/Arithmetic/AddInstruction.h"
#include "PTX/Instructions/Arithmetic/MultiplyWideInstruction.h"
#include "PTX/Instructions/Arithmetic/MADWideInstruction.h"
#include "PTX/Instructions/Comparison/SetPredicateInstruction.h"
#include "PTX/Instructions/ControlFlow/BranchInstruction.h"
#include "PTX/Instructions/Data/MoveAddressInstruction.h"
#include "PTX/Instructions/Data/MoveInstruction.h"
#include "PTX/Instructions/Data/ShuffleInstruction.h"
#include "PTX/Instructions/Data/StoreInstruction.h"
#include "PTX/Instructions/Synchronization/BarrierInstruction.h"
#include "PTX/Instructions/Synchronization/ReductionInstruction.h"
#include "PTX/Operands/Adapters/ArrayAdapter.h"
#include "PTX/Operands/Adapters/PointerAdapter.h"
#include "PTX/Operands/Address/RegisterAddress.h"
#include "PTX/Operands/Variables/AddressableVariable.h"
#include "PTX/Operands/Variables/IndexedRegister.h"
#include "PTX/Operands/Variables/Variable.h"
#include "PTX/Operands/Value.h"
#include "PTX/Statements/CommentStatement.h"
#include "PTX/Statements/Label.h"

namespace Codegen {

enum class ReductionOperation {
	Count,
	Sum,
	Average,
	Minimum,
	Maximum
};

static std::string ReductionOperationString(ReductionOperation reductionOp)
{
	switch (reductionOp)
	{
		case ReductionOperation::Count:
			return "count";
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

	//TODO: Support correct type matrix for reductions

	std::enable_if_t<!std::is_same<T, PTX::PredicateType>::value && T::TypeBits != PTX::Bits::Bits8 && T::TypeBits != PTX::Bits::Bits16, void>
	Generate(const std::string& target, const HorseIR::CallExpression *call) override
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
		if (m_reductionOp == ReductionOperation::Count)
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

		auto value = resources->template AllocateTemporary<T>();

		if (compress == nullptr)
		{
			// If no compression is active, select between the source and the null value depending on the compression

			this->m_builder.AddStatement(new PTX::SelectInstruction<T>(value, src, GenerateNullValue(m_reductionOp), inputPredicate));
		}
		else
		{
			// If the input is within range, select between the source and null values depending on the compression

			auto s1 = new PTX::SelectInstruction<T>(value, src, GenerateNullValue(m_reductionOp), compress);
			s1->SetPredicate(inputPredicate);
			this->m_builder.AddStatement(s1);

			// If the input is out of range, set the index to the null value

			auto s2 = new PTX::MoveInstruction<T>(value, GenerateNullValue(m_reductionOp));
			s2->SetPredicate(inputPredicate, true);
			this->m_builder.AddStatement(s2);
		}

		//TODO: Select which type of reduction we want to implement
		GenerateShuffleBlock(value);
		// GenerateShuffleWarp(value);
		// GenerateShared(value);
	}

	void GenerateShuffleReduction(const PTX::Register<T> *value)
	{
		// Warp shuffle the values down, reducing at each level until a single value is computed, log_2(WARP_SZ)

		auto resources = this->m_builder.GetLocalResources();
		auto warpSize = this->m_builder.GetTargetOptions().WarpSize;

		auto& functionOptions = this->m_builder.GetFunctionOptions();
		functionOptions.SetThreadMultiple(warpSize);

		for (unsigned int offset = warpSize >> 1; offset > 0; offset >>= 1)
		{
			auto temp = resources->template AllocateTemporary<T>();
			this->m_builder.AddStatement(new PTX::ShuffleInstruction<T>(temp, value, new PTX::UInt32Value(offset), new PTX::UInt32Value(warpSize - 1), -1, PTX::ShuffleInstruction<T>::Mode::Down));

			// Generate the operation for the reduction

			auto [result, predicate] = GenerateReductionInstruction(temp, value);

			// (Optionally) move the value into the register used for shuffling

			auto move = new PTX::MoveInstruction<T>(value, result); 
			move->SetPredicate(predicate);
			this->m_builder.AddStatement(move);
		}
	}

	void GenerateShuffleBlock(const PTX::Register<T> *value)
	{
		auto resources = this->m_builder.GetLocalResources();
		auto globalResources = this->m_builder.GetGlobalResources();

		// Allocate shared memory with 1 slot per warp

		//TODO: This is a dynamic variable
		auto sharedMemory = new PTX::ArrayVariableAdapter<T, 32, PTX::SharedSpace>(globalResources->template AllocateSharedMemory<PTX::ArrayType<T, 32>>());

		// Generate the shuffle for each warp

		GenerateShuffleReduction(value);

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

		this->m_builder.AddStatement(new PTX::StoreInstruction<B, T, PTX::SharedSpace, PTX::StoreSynchronization::Volatile>(sharedWarpAddress, value));

		// End the if statement

		this->m_builder.AddStatement(new PTX::BlankStatement());
		this->m_builder.AddStatement(labelWarp);

		// Synchronize all values in shared memory from across warps

		this->m_builder.AddStatement(new PTX::BarrierInstruction(new PTX::UInt32Value(0), true));

		// Load the values back from the shared memory into the individual threads

		auto sharedLaneAddress = addressGenerator.template Generate<T, PTX::SharedSpace>(sharedMemory, AddressGenerator<B>::IndexKind::Lane);
		this->m_builder.AddStatement(new PTX::LoadInstruction<B, T, PTX::SharedSpace, PTX::LoadSynchronization::Volatile>(value, sharedLaneAddress));

		SizeGenerator<B> sizeGen(this->m_builder);
		auto warpCount = sizeGen.GenerateWarpCount();

		auto predActive = resources->template AllocateTemporary<PTX::PredicateType>();
		this->m_builder.AddStatement(new PTX::SetPredicateInstruction<PTX::UInt32Type>(predActive, laneid, warpCount, PTX::UInt32Type::ComparisonOperator::Less));
		this->m_builder.AddStatement(new PTX::SelectInstruction<T>(value, value, GenerateNullValue(m_reductionOp), predActive));

		// Check if we are the first warp in the block for the final part of the reduction

		auto predBlock = resources->template AllocateTemporary<PTX::PredicateType>();
		auto labelBlock = new PTX::Label("RED_BLOCK");
		auto branchBlock = new PTX::BranchInstruction(labelBlock);
		branchBlock->SetPredicate(predBlock);

		auto warpid = indexGen.GenerateWarpIndex();

		this->m_builder.AddStatement(new PTX::SetPredicateInstruction<PTX::UInt32Type>(predWarp, warpid, new PTX::UInt32Value(0), PTX::UInt32Type::ComparisonOperator::NotEqual));
		this->m_builder.AddStatement(branchBlock);
		this->m_builder.AddStatement(new PTX::BlankStatement());

		// Reduce the individual values from all warps into 1 final value for the block

		GenerateShuffleReduction(value);

		// End the if statement

		this->m_builder.AddStatement(new PTX::BlankStatement());
		this->m_builder.AddStatement(labelBlock);

		// Write a single value per thread to the global atomic value

		auto localIndex = indexGen.GenerateLocalIndex();
		GenerateAtomicWrite(localIndex, value);
	}


	void GenerateShuffleWarp(const PTX::Register<T> *value)
	{
		// Generate the shuffle for each warp

		GenerateShuffleReduction(value);

		// Write a single value per thread to the global atomic value

		IndexGenerator indexGen(this->m_builder);
		auto laneid = indexGen.GenerateLaneIndex();

		GenerateAtomicWrite(laneid, value);
	}
	
	void GenerateShared(const PTX::Register<T> *value)
	{
		// Fetch generation state and options

		auto resources = this->m_builder.GetLocalResources();

		auto& targetOptions = this->m_builder.GetTargetOptions();
		auto& inputOptions = this->m_builder.GetInputOptions();

		// Compute the number of threads used in this reduction

		auto& functionOptions = this->m_builder.GetFunctionOptions();

		unsigned int threadCount = targetOptions.MaxBlockSize;
		if (inputOptions.InputSize < threadCount)
		{
			threadCount = std::ceil(std::log2(inputOptions.InputSize));
		}
		functionOptions.SetThreadCount(threadCount);

		auto sharedMemoryAddress = resources->template AllocateSharedMemory<B, T>(threadCount);

		AddressGenerator<B> addressGenerator(this->m_builder);
		auto sharedThreadAddress = addressGenerator.template Generate<T, PTX::SharedSpace>(sharedMemoryAddress->GetVariable(), AddressGenerator<B>::IndexKind::Local, sharedMemoryAddress->GetOffset());

		// Load the the local thread index for accessing the shared memory

		IndexGenerator indexGen(this->m_builder);
		auto localIndex = indexGen.GenerateLocalIndex();

		// Load the initial value into shared memory

		this->m_builder.AddStatement(new PTX::StoreInstruction<B, T, PTX::SharedSpace, PTX::StoreSynchronization::Volatile>(sharedThreadAddress, value));

		// Synchronize the thread group for a consistent view of shared memory

		this->m_builder.AddStatement(new PTX::BarrierInstruction(new PTX::UInt32Value(0), true));

		// Perform an iterative reduction on the shared memory in a pyramid type fashion


		const unsigned int NumThreads = 512;
		int warpSize = targetOptions.WarpSize;

		for (unsigned int i = (NumThreads >> 1); i >= warpSize; i >>= 1)
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
					GenerateSharedReduction(sharedThreadAddress, j);
				}
			}
			else
			{
				GenerateSharedReduction(sharedThreadAddress, i);
			}

			this->m_builder.AddStatement(new PTX::BlankStatement());
			this->m_builder.AddStatement(label);
			if (i > warpSize)
			{
				// If we still have >1 warps running, synchronize the group since they may not be in lock-step

				this->m_builder.AddStatement(new PTX::BarrierInstruction(new PTX::UInt32Value(0), true));
			}
		}

		auto result = resources->template AllocateTemporary<T>();
		auto loadResult = new PTX::LoadInstruction<B, T, PTX::SharedSpace, PTX::LoadSynchronization::Volatile>(result, sharedMemoryAddress);
		GenerateAtomicWrite(localIndex, result, loadResult);
	}

	void GenerateKernelSynchronization(const PTX::Register<T> *value, const PTX::Label *labelEnd)
	{
		// Return 1 value per block

		ReturnGenerator<B> retGenerator(this->m_builder);
		retGenerator.Generate(value, ReturnGenerator<B>::IndexKind::Block);

		//TODO: Implement wave kernel synchronization
	}

	void GenerateAtomicWrite(const PTX::TypedOperand<PTX::UInt32Type> *index, const PTX::Register<T> *value, const PTX::Statement *loadValue = nullptr)
	{
		// At the end of the partial reduction we only have a single active thread. Use it to load the final value

		auto resources = this->m_builder.GetLocalResources();
		auto functionResources = this->m_builder.GetFunctionResources();

		auto& functionOptions = this->m_builder.GetFunctionOptions();
		functionOptions.SetAtomicReturn(true);

		auto predEnd = resources->template AllocateTemporary<PTX::PredicateType>();
		auto labelEnd = new PTX::Label("RED_END");
		auto branchEnd = new PTX::BranchInstruction(labelEnd);
		branchEnd->SetPredicate(predEnd);

		this->m_builder.AddStatement(new PTX::SetPredicateInstruction<PTX::UInt32Type>(predEnd, index, new PTX::UInt32Value(0), PTX::UInt32Type::ComparisonOperator::NotEqual));
		this->m_builder.AddStatement(branchEnd);
		this->m_builder.AddStatement(new PTX::BlankStatement());

		if (loadValue != nullptr)
		{
			this->m_builder.AddStatement(loadValue);
		}

		// Get the function return parameter

		auto returnVariable = functionResources->template GetParameter<PTX::PointerType<B, T>, PTX::ParameterSpace>("$return");

		// Since atomics only output a single value, we use null addressing

		AddressGenerator<B> addressGenerator(this->m_builder);
		auto address = addressGenerator.template GenerateParameter<T, PTX::GlobalSpace>(returnVariable, AddressGenerator<B>::IndexKind::Null);

		// Generate the reduction operation

		this->m_builder.AddStatement(GenerateAtomicInstruction(m_reductionOp, address, value));

		// End the funfction and return

		this->m_builder.AddStatement(new PTX::BlankStatement());
		this->m_builder.AddStatement(labelEnd);
		this->m_builder.AddStatement(new PTX::ReturnInstruction());
	}

private:
	static const PTX::Value<T> *GenerateNullValue(ReductionOperation reductionOp)
	{
		switch (reductionOp)
		{
			case ReductionOperation::Count:
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

	void GenerateSharedReduction(const PTX::Address<B, T, PTX::SharedSpace> *address, unsigned int offset)
	{
		// Load the 2 values for this stage of the reduction into registers using the provided offset. Volatile
		// reads/writes are used to ensure synchronization

		auto resources = this->m_builder.GetLocalResources();

		auto val = resources->template AllocateTemporary<T>();
		auto offsetVal = resources->template AllocateTemporary<T>();
		auto offsetAddress = address->CreateOffsetAddress(offset * PTX::BitSize<T::TypeBits>::NumBytes);

		this->m_builder.AddStatement(new PTX::LoadInstruction<B, T, PTX::SharedSpace, PTX::LoadSynchronization::Volatile>(val, address));
		this->m_builder.AddStatement(new PTX::LoadInstruction<B, T, PTX::SharedSpace, PTX::LoadSynchronization::Volatile>(offsetVal, offsetAddress));

		// Generate the reduction instruction

		auto [result, predicate] = GenerateReductionInstruction(val, offsetVal);

		// (Optionally) store the new value back to the shared memory

		auto store = new PTX::StoreInstruction<B, T, PTX::SharedSpace, PTX::StoreSynchronization::Volatile>(address, result); 
		store->SetPredicate(predicate);
		this->m_builder.AddStatement(store);
	}

	const std::pair<const PTX::Register<T>*, const PTX::Register<PTX::PredicateType> *> GenerateReductionInstruction(const PTX::TypedOperand<T> *val, const PTX::TypedOperand<T> *offsetVal)
	{
		auto resources = this->m_builder.GetLocalResources();
		auto result = resources->template AllocateTemporary<T>();

		// The predicate can (optionally) disable the subsequent store. This is used for comparison type reductions (min/max)

		const PTX::Register<PTX::PredicateType> *predicate = nullptr;
		switch (m_reductionOp)
		{
			case ReductionOperation::Count:
			case ReductionOperation::Average:
			case ReductionOperation::Sum:
				// All 3 reductions are essentially a sum. Count on all 1's, average performs a final division

				this->m_builder.AddStatement(new PTX::AddInstruction<T>(result, val, offsetVal));
				break;
			case ReductionOperation::Minimum:
				// Minimum reduction checks if the offset value is less than the initial value. If so, we
				// will write it back in place of the initial value

				predicate = resources->template AllocateTemporary<PTX::PredicateType>();
				this->m_builder.AddStatement(new PTX::SetPredicateInstruction<T>(predicate, offsetVal, val, T::ComparisonOperator::Less));
				break;
			case ReductionOperation::Maximum:
				// Maximum reduction checks if the offset value is greater than the initial value. If so, we
				// will write it back in place of the initial value

				predicate = resources->template AllocateTemporary<PTX::PredicateType>();
				this->m_builder.AddStatement(new PTX::SetPredicateInstruction<T>(predicate, offsetVal, val, T::ComparisonOperator::Greater));
				break;
			default:
				BuiltinGenerator<B, T>::Unimplemented("reduction operation " + ReductionOperationString(m_reductionOp));
		}

		return std::make_pair(result, predicate);
	}

	static PTX::InstructionStatement *GenerateAtomicInstruction(ReductionOperation reductionOp, const PTX::Address<B, T, PTX::GlobalSpace> *address, const PTX::Register<T> *value)
	{
		switch (reductionOp)
		{
			case ReductionOperation::Count:
			case ReductionOperation::Average:
			case ReductionOperation::Sum:
				return new PTX::ReductionInstruction<B, T, PTX::GlobalSpace, T::ReductionOperation::Add>(address, value);
			case ReductionOperation::Minimum:
				return new PTX::ReductionInstruction<B, T, PTX::GlobalSpace, T::ReductionOperation::Minimum>(address, value);
			case ReductionOperation::Maximum:
				return new PTX::ReductionInstruction<B, T, PTX::GlobalSpace, T::ReductionOperation::Maximum>(address, value);
			default:
				BuiltinGenerator<B, T>::Unimplemented("reduction operation " + ReductionOperationString(reductionOp));
		}
	}

	ReductionOperation m_reductionOp;
};

}
