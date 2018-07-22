#pragma once

#include "Codegen/Generators/Expressions/Builtins/BuiltinGenerator.h"

#include "Codegen/Builder.h"
#include "Codegen/Generators/AddressGenerator.h"
#include "Codegen/Generators/Expressions/OperandGenerator.h"

#include "PTX/StateSpace.h"
#include "PTX/Type.h"
#include "PTX/Declarations/Declaration.h"
#include "PTX/Declarations/SpecialRegisterDeclarations.h"
#include "PTX/Instructions/DevInstruction.h"
#include "PTX/Instructions/Arithmetic/AddInstruction.h"
#include "PTX/Instructions/Arithmetic/MultiplyWideInstruction.h"
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
		// Create a shared memory declaration for the module

		//TODO: Handle multiple shared declarations for a single module (they should use offsets)

		auto sharedDeclaration = new PTX::TypedVariableDeclaration<PTX::ArrayType<T, PTX::DynamicSize>, PTX::SharedSpace>("sdata");
		sharedDeclaration->SetAlignment(PTX::BitSize<T::TypeBits>::NumBytes);
		sharedDeclaration->SetLinkDirective(PTX::Declaration::LinkDirective::External);
		this->m_builder->AddExternalDeclaration(sharedDeclaration);

		auto sharedMemory = new PTX::ArrayVariableAdapter<T, PTX::DynamicSize, PTX::SharedSpace>(sharedDeclaration->GetVariable("sdata"));
		auto sharedMemoryAddress = new PTX::MemoryAddress<B, T, PTX::SharedSpace>(sharedMemory);

		// Generate the address in shared memory for the thread

		AddressGenerator<B> addressGenerator(this->m_builder);
		auto sharedThreadAddress = addressGenerator.template Generate<T, PTX::SharedSpace>(sharedMemory);

		auto srtidx = new PTX::IndexedRegister4<PTX::UInt32Type>(PTX::SpecialRegisterDeclaration_tid->GetVariable("%tid"), PTX::VectorElement::X);
		auto tidx = this->m_builder->template AllocateTemporary<PTX::UInt32Type>("tidx");
		this->m_builder->AddStatement(new PTX::MoveInstruction<PTX::UInt32Type>(tidx, srtidx));

		// Get the initial value for reduction and store it in the shared memory

		if (m_reductionOp == ReductionOperation::Count)
		{
			// A count reduction is value agnostic performs a sum over 1's, one value for each active thread

			auto temp = this->m_builder->template AllocateTemporary<T>();
			this->m_builder->AddStatement(new PTX::MoveInstruction<T>(temp, new PTX::Value<T>(1)));
			this->m_builder->AddStatement(new PTX::StoreInstruction<B, T, PTX::SharedSpace, PTX::StoreSynchronization::Volatile>(sharedThreadAddress, temp));
		}
		else
		{
			// Other reductions combine the values together using some operation

			OperandGenerator<B, T> opGen(this->m_builder);

			// Check if there is a compression on the input value. If so, mask out the initial load
			// according to the predicate

			OperandCompressionGenerator<B, T> compGen(this->m_builder);
			auto compress = compGen.GetCompressionRegister(call->GetArgument(0));

			const PTX::Register<T> *value = nullptr;
			if (compress == nullptr)
			{
				value = opGen.GenerateRegister(call->GetArgument(0));
			}
			else
			{
				auto src = opGen.GenerateOperand(call->GetArgument(0));
				auto temp = this->m_builder->template AllocateTemporary<T>();

				//TODO: Initial value depends on the type of reduction
				this->m_builder->AddStatement(new PTX::SelectInstruction<T>(temp, src, new PTX::Value<T>(0), compress));
				value = temp;
			}

			// Load the initial value into shared memory

			this->m_builder->AddStatement(new PTX::StoreInstruction<B, T, PTX::SharedSpace, PTX::StoreSynchronization::Volatile>(sharedThreadAddress, value));
		}

		// Synchronize the thread group for a consistent view of shared memory

		this->m_builder->AddStatement(new PTX::BarrierInstruction(new PTX::UInt32Value(0), true));

		// Perform an iterative reduction on the shared memory in a pyramid type fashion

		//TODO: Reduce according to the number of elements, not all 512

		const unsigned int WARP_SIZE = 32;
		for (unsigned int i = 256; i >= WARP_SIZE; i >>= 1)
		{
			// At each level, half the threads become inactive

			auto pred = this->m_builder->template AllocateTemporary<PTX::PredicateType>();
			auto guard = new PTX::UInt32Value(i - 1);
			auto label = new PTX::Label("RED_" + std::to_string(i));
			auto branch = new PTX::BranchInstruction(label);
			branch->SetPredicate(pred);

			// Check to see if we are an active thread

			this->m_builder->AddStatement(new PTX::SetPredicateInstruction<PTX::UInt32Type>(pred, tidx, guard, PTX::UInt32Type::ComparisonOperator::Higher));
			this->m_builder->AddStatement(branch);
			this->m_builder->AddStatement(new PTX::BlankStatement());

			if (i == WARP_SIZE)
			{
				// Once the active thread count fits within a warp, reduce without synchronization

				label->SetName("RED_STORE");
				for (unsigned int i = WARP_SIZE; i >= 1; i >>= 1)
				{
					GenerateReduction(sharedThreadAddress, i);
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

		this->m_builder->AddStatement(new PTX::SetPredicateInstruction<PTX::UInt32Type>(predEnd, tidx, new PTX::UInt32Value(0), PTX::UInt32Type::ComparisonOperator::NotEqual));
		this->m_builder->AddStatement(branchEnd);
		this->m_builder->AddStatement(new PTX::BlankStatement());

		this->m_builder->AddStatement(new PTX::LoadInstruction<B, T, PTX::SharedSpace, PTX::LoadSynchronization::Volatile>(target, sharedMemoryAddress));
		this->m_builder->AddStatement(new PTX::BlankStatement());
		this->m_builder->AddStatement(labelEnd);
	}

private:
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
