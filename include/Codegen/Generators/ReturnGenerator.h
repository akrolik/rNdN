#pragma once

#include "Codegen/Generators/Generator.h"

#include "Codegen/Generators/AddressGenerator.h"
#include "Codegen/Generators/Expressions/OperandGenerator.h"
#include "Codegen/Generators/TypeDispatch.h"

#include "HorseIR/Tree/Tree.h"

#include "PTX/PTX.h"

namespace Codegen {

template<PTX::Bits B>
class ReturnGenerator : public Generator
{
public:
	using Generator::Generator;

	using IndexKind = typename AddressGenerator<B>::IndexKind;

	void Generate(const HorseIR::ReturnStatement *returnS)
	{
		auto returnIndex = 0u;
		for (const auto& operand : returnS->GetOperands())
		{
			DispatchType(*this, operand->GetType(), operand, returnIndex++);
		}
	}

	template<class T>
	void Generate(const HorseIR::Operand *operand, unsigned int returnIndex)
	{
		if constexpr(std::is_same<T, PTX::PredicateType>::value)
		{
			// Predicate (1-bit) values are stored as 8 bit integers on the CPU side
			// so a conversion must first be run

			OperandGenerator<B, PTX::PredicateType> opGen(this->m_builder);
			auto value = opGen.GenerateRegister(operand);
			auto converted = ConversionGenerator::ConvertSource<PTX::Int8Type>(this->m_builder, value);
			GenerateWrite(converted, returnIndex);
		}
		else
		{
			OperandGenerator<B, T> opGen(this->m_builder);
			auto value = opGen.GenerateRegister(operand);
			GenerateWrite(value, returnIndex);
		}
	}

	template<class T>
	void GenerateWrite(const PTX::Register<T> *value, unsigned int returnIndex)
	{
		// Check if the register represents a reduction value

		auto resources = this->m_builder.GetLocalResources();
		if (resources->IsReductionRegister(value))
		{
			if constexpr(PTX::is_reduction_type<T>::value)
			{
				IndexGenerator indexGen(this->m_builder);

				auto [granularity, op] = resources->GetReductionRegister(value);
				switch (granularity)
				{
					case RegisterAllocator::ReductionGranularity<T>::Warp:
					{
						auto laneid = indexGen.GenerateLaneIndex();
						GenerateAtomicWrite(laneid, value, op, returnIndex);
						break;
					}
					case RegisterAllocator::ReductionGranularity<T>::Block:
					{
						auto localIndex = indexGen.GenerateLocalIndex();
						GenerateAtomicWrite(localIndex, value, op, returnIndex);
						break;
					}
				}
			}
			else
			{
				//TODO: Support wave reduction for other types
				Utils::Logger::LogError(T::Name() + " does not support atomic reduction");
			}
		}
		else
		{
			//TODO: Use shape analysis for loading the correct index
			GenerateIndexedWrite(value, returnIndex, IndexKind::Global);
		}
	}

	// template<class T, typename Enable = std::enable_if_t<PTX::StoreInstruction<B, T, PTX::GlobalSpace, PTX::StoreSynchronization::Weak, false>::TypeSupported>>
	template<class T>
	void GenerateIndexedWrite(const PTX::Register<T> *value, unsigned int returnIndex, IndexKind indexKind)
	{
		// Fetch the return variable

		auto returnName = "$return_" + std::to_string(returnIndex);
		auto kernelResources = this->m_builder.GetKernelResources();
		auto variable = kernelResources->template GetParameter<PTX::PointerType<B, T>, PTX::ParameterSpace>(returnName);

		// Store the value at the appropriate index

		AddressGenerator<B> addressGenerator(this->m_builder);
		auto address = addressGenerator.template GenerateParameter<T, PTX::GlobalSpace>(variable, indexKind);
		this->m_builder.AddStatement(new PTX::StoreInstruction<B, T, PTX::GlobalSpace>(address, value));
	}

	template<class T>
	void GenerateAtomicWrite(const PTX::TypedOperand<PTX::UInt32Type> *index, const PTX::Register<T> *value, RegisterAllocator::ReductionOperation<T> reductionOp, unsigned int returnIndex)
	{
		// At the end of the partial reduction we only have a single active thread. Use it to load the final value

		auto resources = this->m_builder.GetLocalResources();
		auto kernelResources = this->m_builder.GetKernelResources();

		auto predicate = resources->template AllocateTemporary<PTX::PredicateType>();
		auto label = new PTX::Label("RET_" + std::to_string(returnIndex));
		auto branch = new PTX::BranchInstruction(label);
		branch->SetPredicate(predicate);

		this->m_builder.AddStatement(new PTX::SetPredicateInstruction<PTX::UInt32Type>(predicate, index, new PTX::UInt32Value(0), PTX::UInt32Type::ComparisonOperator::NotEqual));
		this->m_builder.AddStatement(branch);
		this->m_builder.AddStatement(new PTX::BlankStatement());

		// Get the function return parameter

		auto returnName = "$return_" + std::to_string(returnIndex);
		auto returnVariable = kernelResources->template GetParameter<PTX::PointerType<B, T>, PTX::ParameterSpace>(returnName);

		// Since atomics only output a single value, we use null addressing

		AddressGenerator<B> addressGenerator(this->m_builder);
		auto address = addressGenerator.template GenerateParameter<T, PTX::GlobalSpace>(returnVariable, AddressGenerator<B>::IndexKind::Null);

		// Generate the reduction operation

		this->m_builder.AddStatement(GenerateAtomicInstruction(reductionOp, address, value));

		// End the funfction and return

		this->m_builder.AddStatement(new PTX::BlankStatement());
		this->m_builder.AddStatement(label);
	}

	template<class T>
	static PTX::InstructionStatement *GenerateAtomicInstruction(RegisterAllocator::ReductionOperation<T> reductionOp, const PTX::Address<B, T, PTX::GlobalSpace> *address, const PTX::Register<T> *value)
	{
		switch (reductionOp)
		{
			case RegisterAllocator::ReductionOperation<T>::Add:
				return new PTX::ReductionInstruction<B, T, PTX::GlobalSpace, T::ReductionOperation::Add>(address, value);
			case RegisterAllocator::ReductionOperation<T>::Minimum:
				return new PTX::ReductionInstruction<B, T, PTX::GlobalSpace, T::ReductionOperation::Minimum>(address, value);
			case RegisterAllocator::ReductionOperation<T>::Maximum:
				return new PTX::ReductionInstruction<B, T, PTX::GlobalSpace, T::ReductionOperation::Maximum>(address, value);
			default:
				Utils::Logger::LogError("Generator does not support reduction operation");
		}
	}
};

}
