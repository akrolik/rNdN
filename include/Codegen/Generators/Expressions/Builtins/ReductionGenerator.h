#pragma once

#include <limits>

#include "Codegen/Generators/Expressions/Builtins/BuiltinGenerator.h"

#include "Codegen/Builder.h"
#include "Codegen/Generators/AddressGenerator.h"
#include "Codegen/Generators/IndexGenerator.h"
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
#include "PTX/Instructions/Data/StoreInstruction.h"
#include "PTX/Instructions/Synchronization/BarrierInstruction.h"
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
	ReductionGenerator(Builder *builder, ReductionOperation reductionOp) : BuiltinGenerator<B, T>(builder), m_reductionOp(reductionOp) {}

	// The output of a reduction function has no compression predicate.
	// We therefore do not implement GenerateCompressionPredicate in this subclass

	//TODO: Support 8 bit types in reductions
	//TODO: Investigate using a shuffle reduction instead of the typical shared memory

	std::enable_if_t<!std::is_same<T, PTX::PredicateType>::value && T::TypeBits != PTX::Bits::Bits8, void>
	Generate(const PTX::Register<T> *target, const HorseIR::CallExpression *call) override
	{
		// Load the underlying data size from the kernel parameters

		const std::string sizeName = "$size";
		auto sizeDeclaration = new PTX::TypedVariableDeclaration<PTX::UInt64Type, PTX::ParameterSpace>(sizeName);
		this->m_builder->AddParameter(sizeDeclaration);
		auto sizeVariable = sizeDeclaration->GetVariable(sizeName);

		auto sizeAddress = new PTX::MemoryAddress<B, PTX::UInt64Type, PTX::ParameterSpace>(sizeVariable);
		auto size = this->m_builder->template AllocateTemporary<PTX::UInt64Type>("size");
		this->m_builder->AddStatement(new PTX::LoadInstruction<B, PTX::UInt64Type, PTX::ParameterSpace>(size, sizeAddress));

		// Generate the address in shared memory for the thread

		auto sharedMemoryAddress = this->m_builder->template AllocateSharedMemory<B, T>(512);

		AddressGenerator<B> addressGenerator(this->m_builder);
		auto sharedThreadAddress = addressGenerator.template Generate<T, PTX::SharedSpace>(sharedMemoryAddress->GetVariable(), sharedMemoryAddress->GetOffset());

		// Load the thread indexes

		IndexGenerator indexGen(this->m_builder);
		auto localIndex = indexGen.GenerateLocalIndex();
		auto globalIndex = indexGen.GenerateGlobalIndex();

		// Some operations require a 64-bit value for comparison

		auto globalIndex64 = this->m_builder->template AllocateTemporary<PTX::UInt64Type>();
		this->m_builder->AddStatement(new PTX::ConvertInstruction<PTX::UInt64Type, PTX::UInt32Type>(globalIndex64, globalIndex));

		// Check if the thread is in bounds for the input data

		auto inputPredicate = this->m_builder->template AllocateTemporary<PTX::PredicateType>();
		this->m_builder->AddStatement(new PTX::SetPredicateInstruction<PTX::UInt64Type>(inputPredicate, globalIndex64, size, PTX::UInt64Type::ComparisonOperator::Less));

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

		auto value = this->m_builder->template AllocateTemporary<T>();

		if (compress == nullptr)
		{
			// If no compression is active, select between the source and the null value depending on the compression

			this->m_builder->AddStatement(new PTX::SelectInstruction<T>(value, src, GenerateNullValue(m_reductionOp), inputPredicate));
		}
		else
		{
			// If the input is within range, select between the source and null values depending on the compression

			auto s1 = new PTX::SelectInstruction<T>(value, src, GenerateNullValue(m_reductionOp), compress);
			s1->SetPredicate(inputPredicate);
			this->m_builder->AddStatement(s1);

			// If the input is out of range, set the index to the null value

			auto s2 = new PTX::MoveInstruction<T>(value, GenerateNullValue(m_reductionOp));
			s2->SetPredicate(inputPredicate, true);
			this->m_builder->AddStatement(s2);
		}

		// Load the initial value into shared memory

		this->m_builder->AddStatement(new PTX::StoreInstruction<B, T, PTX::SharedSpace, PTX::StoreSynchronization::Volatile>(sharedThreadAddress, value));

		// Synchronize the thread group for a consistent view of shared memory

		this->m_builder->AddStatement(new PTX::BarrierInstruction(new PTX::UInt32Value(0), true));

		// Perform an iterative reduction on the shared memory in a pyramid type fashion

		const unsigned int NumThreads = 512;
		const unsigned int WARP_SIZE = 32;

		for (unsigned int i = (NumThreads >> 1); i >= WARP_SIZE; i >>= 1)
		{
			// At each level, half the threads become inactive

			auto pred = this->m_builder->template AllocateTemporary<PTX::PredicateType>();
			auto guard = new PTX::UInt32Value(i - 1);
			auto label = new PTX::Label("RED_" + std::to_string(i));
			auto branch = new PTX::BranchInstruction(label);
			branch->SetPredicate(pred);

			// Check to see if we are an active thread

			this->m_builder->AddStatement(new PTX::SetPredicateInstruction<PTX::UInt32Type>(pred, localIndex, guard, PTX::UInt32Type::ComparisonOperator::Higher));
			this->m_builder->AddStatement(branch);
			this->m_builder->AddStatement(new PTX::BlankStatement());

			if (i <= WARP_SIZE)
			{
				// Once the active thread count fits within a warp, reduce without synchronization

				label->SetName("RED_STORE");
				for (unsigned int j = i; j >= 1; j >>= 1)
				{
					GenerateReduction(sharedThreadAddress, j);
				}
			}
			else
			{
				GenerateReduction(sharedThreadAddress, i);
			}

			this->m_builder->AddStatement(new PTX::BlankStatement());
			this->m_builder->AddStatement(label);
			if (i > WARP_SIZE)
			{
				// If we still have >1 warps running, synchronize the group since they may not be in lock-step

				this->m_builder->AddStatement(new PTX::BarrierInstruction(new PTX::UInt32Value(0), true));
			}
		}

		// At the end of the partial reduction we only have a single active thread. Use it to load the final value

		auto predEnd = this->m_builder->template AllocateTemporary<PTX::PredicateType>();
		auto labelEnd = new PTX::Label("RED_END");
		auto branchEnd = new PTX::BranchInstruction(labelEnd);
		branchEnd->SetPredicate(predEnd);

		this->m_builder->AddStatement(new PTX::SetPredicateInstruction<PTX::UInt32Type>(predEnd, localIndex, new PTX::UInt32Value(0), PTX::UInt32Type::ComparisonOperator::NotEqual));
		this->m_builder->AddStatement(branchEnd);
		this->m_builder->AddStatement(new PTX::BlankStatement());

		this->m_builder->AddStatement(new PTX::LoadInstruction<B, T, PTX::SharedSpace, PTX::LoadSynchronization::Volatile>(target, sharedMemoryAddress));
		this->m_builder->AddStatement(new PTX::BlankStatement());
		this->m_builder->AddStatement(labelEnd);
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

	void GenerateReduction(const PTX::Address<B, T, PTX::SharedSpace> *address, unsigned int offset)
	{
		// Load the 2 values for this stage of the reduction into registers using the provided offset. Volatile
		// reads/writes are used to ensure synchronization

		auto result = this->m_builder->template AllocateTemporary<T>();
		auto val = this->m_builder->template AllocateTemporary<T>();
		auto offsetVal = this->m_builder->template AllocateTemporary<T>();
		auto offsetAddress = address->CreateOffsetAddress(offset * PTX::BitSize<T::TypeBits>::NumBytes);

		this->m_builder->AddStatement(new PTX::LoadInstruction<B, T, PTX::SharedSpace, PTX::LoadSynchronization::Volatile>(val, address));
		this->m_builder->AddStatement(new PTX::LoadInstruction<B, T, PTX::SharedSpace, PTX::LoadSynchronization::Volatile>(offsetVal, offsetAddress));

		// The predicate can (optionally) disable the subsequent store. This is used for comparison
		// type reductions (min/max)

		const PTX::Register<PTX::PredicateType> *predicate = nullptr;
		switch (m_reductionOp)
		{
			case ReductionOperation::Count:
			case ReductionOperation::Average:
			case ReductionOperation::Sum:
				// All 3 reductions are essentially a sum. Count on all 1's, average performs a final division

				this->m_builder->AddStatement(new PTX::AddInstruction<T>(result, val, offsetVal));
				break;
			case ReductionOperation::Minimum:
				// Minimum reduction checks if the offset value is less than the initial value. If so, we
				// will write it back in place of the initial value

				predicate = this->m_builder->template AllocateTemporary<PTX::PredicateType>();
				this->m_builder->AddStatement(new PTX::SetPredicateInstruction<T>(predicate, offsetVal, val, T::ComparisonOperator::Less));
				break;
			case ReductionOperation::Maximum:
				// Maximum reduction checks if the offset value is greater than the initial value. If so, we
				// will write it back in place of the initial value

				predicate = this->m_builder->template AllocateTemporary<PTX::PredicateType>();
				this->m_builder->AddStatement(new PTX::SetPredicateInstruction<T>(predicate, offsetVal, val, T::ComparisonOperator::Greater));
				break;
			default:
				BuiltinGenerator<B, T>::Unimplemented("reduction operation " + ReductionOperationString(m_reductionOp));
		}

		// (Optionally) store the new value back to the shared memory

		auto store = new PTX::StoreInstruction<B, T, PTX::SharedSpace, PTX::StoreSynchronization::Volatile>(address, result); 
		store->SetPredicate(predicate);
		this->m_builder->AddStatement(store);
	}

	ReductionOperation m_reductionOp;
};

}
