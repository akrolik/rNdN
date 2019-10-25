#pragma once

#include "Codegen/Generators/Generator.h"

#include "Codegen/Builder.h"
#include "Codegen/Generators/AddressGenerator.h"
#include "Codegen/Generators/IndexGenerator.h"
#include "Codegen/Generators/Expressions/OperandGenerator.h"
#include "Codegen/Generators/TypeDispatch.h"

#include "HorseIR/Tree/Tree.h"

#include "PTX/PTX.h"

namespace Codegen {

template<PTX::Bits B>
class ValueStoreGenerator : public Generator
{
public:
	using Generator::Generator;

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
			// Predicate (1-bit) values are stored as 8 bit integers on the CPU side so a conversion must first be run

			Generate<PTX::Int8Type>(operand, returnIndex);
		}
		else
		{
			GenerateWrite<T>(operand, returnIndex);
		}
	}

	//TODO: Ensure that when we compress/move registers, we do not lose the compression or reduction flags

	template<class T>
	void GenerateWrite(const HorseIR::Operand *operand, unsigned int returnIndex)
	{
		auto& inputOptions = this->m_builder.GetInputOptions();

		// Select the write kind based on the thread geometry and return shape

		auto shape = inputOptions.ReturnShapes.at(returnIndex);
		if (const auto vectorGeometry = Analysis::ShapeUtils::GetShape<Analysis::VectorShape>(inputOptions.ThreadGeometry))
		{
			if (const auto vectorShape = Analysis::ShapeUtils::GetShape<Analysis::VectorShape>(shape))
			{
				// Check for the style of write:
				//  (1) Reduction (we assume this corresponds to scalar output in a non-scalar kernel)
				//  (2) Vector
				//  (3) Compression

				if (Analysis::ShapeUtils::IsScalarSize(vectorShape->GetSize()) && !Analysis::ShapeUtils::IsScalarSize(vectorGeometry->GetSize()))
				{
					GenerateWriteReduction<T>(operand, OperandGenerator<B, T>::LoadKind::Vector, IndexGenerator::Kind::Null, returnIndex);
					return;
				}
				else if (*vectorGeometry == *vectorShape)
				{
					GenerateWriteVector<T>(operand, IndexGenerator::Kind::Global, returnIndex);
					return;
				}
				else if (Analysis::ShapeUtils::IsSize<Analysis::Shape::CompressedSize>(vectorShape->GetSize()))
				{
					GenerateWriteCompressed<T>(operand, returnIndex);
					return;
				}
			}
		}
		else if (const auto listGeometry = Analysis::ShapeUtils::GetShape<Analysis::ListShape>(inputOptions.ThreadGeometry))
		{
			if (const auto vectorShape = Analysis::ShapeUtils::GetShape<Analysis::VectorShape>(shape))
			{
				// Special horizontal write for @raze function

				GenerateWriteReduction<T>(operand, OperandGenerator<B, T>::LoadKind::ListCell, IndexGenerator::Kind::Cell, returnIndex);
				return;
			}
			else if (const auto listShape = Analysis::ShapeUtils::GetShape<Analysis::ListShape>(shape))
			{
				// Check for the style of write:
				//  (1) Reduction (we assume this corresponds to scalar output in a non-scalar cell)
				//  (2) List

				const auto cellShape = Analysis::ShapeUtils::MergeShapes(listShape->GetElementShapes());
				const auto cellGeometry = Analysis::ShapeUtils::MergeShapes(listGeometry->GetElementShapes());

				const auto cellVector = Analysis::ShapeUtils::GetShape<Analysis::VectorShape>(cellShape);
				const auto cellVectorGeometry = Analysis::ShapeUtils::GetShape<Analysis::VectorShape>(cellGeometry);

				if (cellVector && cellVectorGeometry && Analysis::ShapeUtils::IsScalarSize(cellVector->GetSize()) && !Analysis::ShapeUtils::IsScalarSize(cellVectorGeometry->GetSize()))
				{
					GenerateWriteReduction<T>(operand, OperandGenerator<B, T>::LoadKind::Vector, IndexGenerator::Kind::Null, returnIndex);
					return;
				}
				else if (*listGeometry == *listShape)
				{
					GenerateWriteVector<T>(operand, IndexGenerator::Kind::CellData, returnIndex);
					return;
				}
			}
		}
		Utils::Logger::LogError("Unable to generate store for thread geometry " + Analysis::ShapeUtils::ShapeString(inputOptions.ThreadGeometry));
	}

	template<class T>
	void GenerateWriteReduction(const HorseIR::Operand *operand, typename OperandGenerator<B, T>::LoadKind loadKind, IndexGenerator::Kind indexKind, unsigned int returnIndex)
	{
		// Check if the register represents a reduction value

		if constexpr(PTX::is_reduction_type<T>::value)
		{
			auto resources = this->m_builder.GetLocalResources();

			OperandGenerator<B, T> operandGenerator(this->m_builder);
			auto value = operandGenerator.GenerateRegister(operand, loadKind);
			auto [granularity, op] = resources->GetReductionRegister(value);

			IndexGenerator indexGenerator(this->m_builder);
			auto writeIndex = indexGenerator.GenerateIndex(indexKind);

			switch (granularity)
			{
				case RegisterAllocator::ReductionGranularity<T>::Warp:
				{
					IndexGenerator indexGenerator(this->m_builder);
					auto laneid = indexGenerator.GenerateLaneIndex();
					GenerateAtomicWrite(laneid, value, writeIndex, op, returnIndex);
					break;
				}
				case RegisterAllocator::ReductionGranularity<T>::Block:
				{
					IndexGenerator indexGen(this->m_builder);
					auto localIndex = indexGen.GenerateLocalIndex();
					GenerateAtomicWrite(localIndex, value, writeIndex, op, returnIndex);
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

	template<class T>
	void GenerateAtomicWrite(const PTX::TypedOperand<PTX::UInt32Type> *activeIndex, const PTX::Register<T> *value, const PTX::TypedOperand<PTX::UInt32Type> *writeIndex, RegisterAllocator::ReductionOperation<T> reductionOp, unsigned int returnIndex)
	{
		// At the end of the partial reduction we only have a single active thread. Use it to store the final value

		auto resources = this->m_builder.GetLocalResources();
		auto kernelResources = this->m_builder.GetKernelResources();

		auto predicate = resources->template AllocateTemporary<PTX::PredicateType>();
		auto label = new PTX::Label("RET_" + std::to_string(returnIndex));
		auto branch = new PTX::BranchInstruction(label);
		branch->SetPredicate(predicate);

		this->m_builder.AddStatement(new PTX::SetPredicateInstruction<PTX::UInt32Type>(predicate, activeIndex, new PTX::UInt32Value(0), PTX::UInt32Type::ComparisonOperator::NotEqual));
		this->m_builder.AddStatement(branch);
		this->m_builder.AddStatement(new PTX::BlankStatement());

		// Get the address of the write location depending on the indexing kind and the reduction operation

		auto address = GenerateAddress<T>(returnIndex, writeIndex);
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

	template<class T>
	void GenerateWriteVector(const HorseIR::Operand *operand, IndexGenerator::Kind indexKind, unsigned int returnIndex)
	{
		// Fetch the write address and store the value at the appropriate index (global indexing)

		OperandGenerator<B, T> operandGenerator(this->m_builder);
		auto value = operandGenerator.GenerateRegister(operand, OperandGenerator<B, T>::LoadKind::Vector);

		IndexGenerator indexGenerator(this->m_builder);
		auto index = indexGenerator.GenerateIndex(indexKind);

		auto address = GenerateAddress<T>(returnIndex, index);
		this->m_builder.AddStatement(new PTX::StoreInstruction<B, T, PTX::GlobalSpace>(address, value));
	}

	template<class T>
	void GenerateWriteCompressed(const HorseIR::Operand *operand, unsigned int returnIndex)
	{
		//TODO: Compressed output
	}

	template<class T>
	PTX::Address<B, T, PTX::GlobalSpace> *GenerateAddress(unsigned int returnIndex, const PTX::TypedOperand<PTX::UInt32Type> *index)
	{
		// Get the address register for the return and offset by the index

		auto resources = this->m_builder.GetLocalResources();

		auto name = NameUtils::DataAddressName(NameUtils::ReturnName(returnIndex));
		auto addressRegister = resources->GetRegister<PTX::UIntType<B>>(name);

		// Generate the address for the correct index

		AddressGenerator<B> addressGenerator(this->m_builder);
		return addressGenerator.template GenerateAddress<T, PTX::GlobalSpace>(addressRegister, index);
	}

};

}
