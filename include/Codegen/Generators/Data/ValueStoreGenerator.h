#pragma once

#include <string>

#include "Codegen/Generators/Generator.h"

#include "Analysis/Shape/Shape.h"
#include "Analysis/Shape/ShapeUtils.h"

#include "Codegen/Builder.h"
#include "Codegen/Generators/AddressGenerator.h"
#include "Codegen/Generators/AtomicGenerator.h"
#include "Codegen/Generators/IndexGenerator.h"
#include "Codegen/Generators/PrefixSumGenerator.h"
#include "Codegen/Generators/TypeDispatch.h"
#include "Codegen/Generators/Expressions/OperandGenerator.h"

#include "HorseIR/Tree/Tree.h"

#include "PTX/PTX.h"

#include "Utils/Logger.h"

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

				if (cellVector && cellVectorGeometry)
				{
					if (Analysis::ShapeUtils::IsScalarSize(cellVector->GetSize()) && !Analysis::ShapeUtils::IsScalarSize(cellVectorGeometry->GetSize()))
					{
						GenerateWriteReduction<T>(operand, OperandGenerator<B, T>::LoadKind::Vector, IndexGenerator::Kind::Null, returnIndex);
						return;
					}
					else if (*cellVector == *cellVectorGeometry)
					{
						GenerateWriteVector<T>(operand, IndexGenerator::Kind::CellData, returnIndex);
						return;
					}
					else if (Analysis::ShapeUtils::IsSize<Analysis::Shape::CompressedSize>(cellVector->GetSize()))
					{
						GenerateWriteCompressed<T>(operand, returnIndex);
						return;
					}
				}
			}
		}
		Utils::Logger::LogError("Unable to generate store for shape " + Analysis::ShapeUtils::ShapeString(shape) + " in thread geometry " + Analysis::ShapeUtils::ShapeString(inputOptions.ThreadGeometry));
	}

	template<class T>
	void GenerateWriteReduction(const HorseIR::Operand *operand, typename OperandGenerator<B, T>::LoadKind loadKind, IndexGenerator::Kind indexKind, unsigned int returnIndex)
	{
		auto resources = this->m_builder.GetLocalResources();

		// Get the reduction properties for the register and the write index

		OperandGenerator<B, T> operandGenerator(this->m_builder);
		auto value = operandGenerator.GenerateRegister(operand, loadKind);
		auto [granularity, op] = resources->GetReductionRegister(value);

		IndexGenerator indexGenerator(this->m_builder);
		auto writeIndex = indexGenerator.GenerateIndex(indexKind);

		// Generate the reduction value depending on the reduced granularity

		switch (granularity)
		{
			case RegisterReductionGranularity::Warp:
			{
				IndexGenerator indexGenerator(this->m_builder);
				auto laneid = indexGenerator.GenerateLaneIndex();
				GenerateWriteReduction(op, laneid, value, writeIndex, returnIndex);
				break;
			}
			case RegisterReductionGranularity::Block:
			{
				IndexGenerator indexGen(this->m_builder);
				auto localIndex = indexGen.GenerateLocalIndex();
				GenerateWriteReduction(op, localIndex, value, writeIndex, returnIndex);
				break;
			}
		}
	}

	template<class T>
	void GenerateWriteReduction(RegisterReductionOperation reductionOp, const PTX::TypedOperand<PTX::UInt32Type> *activeIndex, const PTX::Register<T> *value, const PTX::TypedOperand<PTX::UInt32Type> *writeIndex, unsigned int returnIndex)
	{
		// At the end of the partial reduction we only have a single active thread. Use it to store the final value

		auto resources = this->m_builder.GetLocalResources();

		auto predicate = resources->template AllocateTemporary<PTX::PredicateType>();
		auto label = this->m_builder.CreateLabel("RET_" + std::to_string(returnIndex));

		this->m_builder.AddStatement(new PTX::SetPredicateInstruction<PTX::UInt32Type>(predicate, activeIndex, new PTX::UInt32Value(0), PTX::UInt32Type::ComparisonOperator::NotEqual));
		this->m_builder.AddStatement(new PTX::BranchInstruction(label, predicate));
		this->m_builder.AddStatement(new PTX::BlankStatement());

		// Get the address of the write location depending on the indexing kind and the reduction operation

		auto address = GenerateAddress<T>(returnIndex, writeIndex);
		GenerateWriteReduction(reductionOp, address, value, returnIndex);

		// End the function and return

		this->m_builder.AddStatement(new PTX::BlankStatement());
		this->m_builder.AddStatement(label);
	}

	template<class T>
	void GenerateWriteReduction(RegisterReductionOperation reductionOp, const PTX::Address<B, T, PTX::GlobalSpace> *address, const PTX::Register<T> *value, unsigned int returnIndex)
	{
		if constexpr(PTX::is_reduction_type<T>::value)
		{
			GenerateReductionInstruction(reductionOp, address, value, returnIndex);
		}
		else
		{
			GenerateCASWriteReduction(reductionOp, address, value, returnIndex);
		}
	}

	template<class T>
	void GenerateReductionInstruction(RegisterReductionOperation reductionOp, const PTX::Address<B, T, PTX::GlobalSpace> *address, const PTX::Register<T> *value, unsigned int returnIndex)
	{
		switch (reductionOp)
		{
			case RegisterReductionOperation::Add:
			{
				if constexpr(std::is_same<T, PTX::Int64Type>::value)
				{
					GenerateCASWriteReduction(reductionOp, address, value, returnIndex);
				}
				else
				{
					this->m_builder.AddStatement(new PTX::ReductionInstruction<B, T, PTX::GlobalSpace, T::ReductionOperation::Add>(address, value));
				}
				break;
			}
			case RegisterReductionOperation::Minimum:
			{
				this->m_builder.AddStatement(new PTX::ReductionInstruction<B, T, PTX::GlobalSpace, T::ReductionOperation::Minimum>(address, value));
				break;
			}
			case RegisterReductionOperation::Maximum:
			{
				this->m_builder.AddStatement(new PTX::ReductionInstruction<B, T, PTX::GlobalSpace, T::ReductionOperation::Maximum>(address, value));
				break;
			}
			default:
			{
				Utils::Logger::LogError("Store generator does not support reduction operation");
			}
		}
	}

	template<class T>
	void GenerateCASWriteReduction(RegisterReductionOperation reductionOp, const PTX::Address<B, T, PTX::GlobalSpace> *address, const PTX::Register<T> *value, unsigned int returnIndex)
	{
		auto resources = this->m_builder.GetLocalResources();

		// Generate lock variable and lock

		auto globalResources = this->m_builder.GetGlobalResources();
		auto lock = globalResources->template AllocateGlobalVariable<PTX::Bit32Type>(NameUtils::ReturnName(returnIndex) + "_lock");

		AtomicGenerator<B> atomicGenerator(this->m_builder);
		atomicGenerator.GenerateWait(lock);

		// Load the existing value

		auto global = resources->template AllocateTemporary<T>();
		this->m_builder.AddStatement(new PTX::LoadInstruction<B, T, PTX::GlobalSpace>(global, address));

		// Generate the reduction

		auto predicate = GenerateCASReductionOperation(reductionOp, global, value);

		if (predicate)
		{
			// If we have a predicate, store the new value if needed only

			auto store = new PTX::StoreInstruction<B, T, PTX::GlobalSpace>(address, value);
			store->SetPredicate(predicate);
			this->m_builder.AddStatement(store);
		}
		else
		{
			// If no predicate, we assume the global has been updated with the new value

			this->m_builder.AddStatement(new PTX::StoreInstruction<B, T, PTX::GlobalSpace>(address, global));
		}

		// Unlock!

		atomicGenerator.GenerateUnlock(lock);
	}

	template<class T>
	const PTX::Register<PTX::PredicateType> *GenerateCASReductionOperation(RegisterReductionOperation reductionOp, const PTX::Register<T> *global, const PTX::Register<T> *value)
	{
		if constexpr(std::is_same<T, PTX::Int8Type>::value)
		{
			auto convertedGlobal = ConversionGenerator::ConvertSource<PTX::Int16Type, T>(this->m_builder, global);
			auto convertedValue = ConversionGenerator::ConvertSource<PTX::Int16Type, T>(this->m_builder, value);

			auto predicate = GenerateCASReductionOperation(reductionOp, convertedGlobal, convertedValue);
			if (predicate)
			{
				return predicate;
			}

			ConversionGenerator::ConvertSource<T, PTX::Int16Type>(this->m_builder, global, convertedGlobal);
			return nullptr;
		}
		else
		{
			auto resources = this->m_builder.GetLocalResources();
			switch (reductionOp)
			{
				case RegisterReductionOperation::Add:
				{
					this->m_builder.AddStatement(new PTX::AddInstruction<T>(global, global, value));
					return nullptr;
				}
				case RegisterReductionOperation::Minimum:
				{
					auto predicate = resources->template AllocateTemporary<PTX::PredicateType>();
					this->m_builder.AddStatement(new PTX::SetPredicateInstruction<T>(predicate, global, value, T::ComparisonOperator::Less));
					return predicate;
				}
				case RegisterReductionOperation::Maximum:
				{
					auto predicate = resources->template AllocateTemporary<PTX::PredicateType>();
					this->m_builder.AddStatement(new PTX::SetPredicateInstruction<T>(predicate, global, value, T::ComparisonOperator::Greater));
					return predicate;
				}
				default:
				{
					Utils::Logger::LogError("Store generator does not support CAS operation");
				}
			}
		}
	}

	template<class T>
	void GenerateWriteVector(const HorseIR::Operand *operand, IndexGenerator::Kind indexKind, unsigned int returnIndex)
	{
		// Fetch the write address and store the value at the appropriate index (global indexing, or indexed register)

		OperandGenerator<B, T> operandGenerator(this->m_builder);
		auto value = operandGenerator.GenerateRegister(operand, OperandGenerator<B, T>::LoadKind::Vector);

		auto resources = this->m_builder.GetLocalResources();
		if (auto indexed = resources->template GetIndexedRegister<T>(value))
		{
			auto address = GenerateAddress<T>(returnIndex, indexed);
			this->m_builder.AddStatement(new PTX::StoreInstruction<B, T, PTX::GlobalSpace>(address, value));
		}
		else
		{
			// Global/cell data indexing depending on thread geometry

			IndexGenerator indexGenerator(this->m_builder);
			auto index = indexGenerator.GenerateIndex(indexKind);

			auto address = GenerateAddress<T>(returnIndex, index);
			this->m_builder.AddStatement(new PTX::StoreInstruction<B, T, PTX::GlobalSpace>(address, value));
		}
	}

	template<class T>
	void GenerateWriteCompressed(const HorseIR::Operand *operand, unsigned int returnIndex)
	{
		auto resources = this->m_builder.GetLocalResources();

		// Fetch the data and write to the compressed index

		OperandGenerator<B, T> operandGenerator(this->m_builder);
		auto value = operandGenerator.GenerateRegister(operand, OperandGenerator<B, T>::LoadKind::Vector);

		if (auto predicate = resources->template GetCompressedRegister<T>(value))
		{
			AddressGenerator<B> addressGenerator(this->m_builder);

			// Generate the in-order global prefix sum! Convert the predicate to integer values for the sum

			auto intPredicate = resources->template AllocateTemporary<PTX::UInt32Type>();
			this->m_builder.AddStatement(new PTX::SelectInstruction<PTX::UInt32Type>(intPredicate, new PTX::UInt32Value(1), new PTX::UInt32Value(0), predicate));

			// Calculate prefix sum

			auto kernelResources = this->m_builder.GetKernelResources();
			auto parameter = kernelResources->GetParameter<PTX::PointerType<B, T>>(NameUtils::ReturnName(returnIndex));

			auto sizeParameter = kernelResources->GetParameter<PTX::PointerType<B, PTX::UInt32Type>>(NameUtils::SizeName(parameter));
			auto sizeAddress = addressGenerator.template GenerateAddress<PTX::UInt32Type>(sizeParameter);

			PrefixSumGenerator<B> prefixSumGenerator(this->m_builder);
			auto writeIndex = prefixSumGenerator.template Generate<PTX::UInt32Type>(sizeAddress, intPredicate, PrefixSumMode::Exclusive);

			// Check for compression - this will mask outputs

			auto label = this->m_builder.CreateLabel("RET_" + std::to_string(returnIndex));

			this->m_builder.AddStatement(new PTX::BranchInstruction(label, predicate, true));
			this->m_builder.AddStatement(new PTX::BlankStatement());

			// Store the value at the place specified by the prefix sum

			auto address = GenerateAddress<T>(returnIndex, writeIndex);
			this->m_builder.AddStatement(new PTX::StoreInstruction<B, T, PTX::GlobalSpace>(address, value));

			this->m_builder.AddStatement(new PTX::BlankStatement());
			this->m_builder.AddStatement(label);
		}
		else
		{
			Utils::Logger::LogError("Unable to find compression predicate for return parameter " + NameUtils::ReturnName(returnIndex));
		}
	}

	template<class T>
	PTX::Address<B, T, PTX::GlobalSpace> *GenerateAddress(unsigned int returnIndex, const PTX::TypedOperand<PTX::UInt32Type> *index)
	{
		// Get the address register for the return and offset by the index

		auto kernelResources = this->m_builder.GetKernelResources();
		auto returnName = NameUtils::ReturnName(returnIndex);

		AddressGenerator<B> addressGenerator(this->m_builder);

		auto& inputOptions = this->m_builder.GetInputOptions();
		auto shape = inputOptions.ReturnShapes.at(returnIndex);
		if (Analysis::ShapeUtils::IsShape<Analysis::VectorShape>(shape))
		{
			auto returnParameter = kernelResources->template GetParameter<PTX::PointerType<B, T>>(returnName);
			return addressGenerator.template GenerateAddress<T>(returnParameter, index);
		}
		else if (Analysis::ShapeUtils::IsShape<Analysis::ListShape>(shape))
		{
			auto returnParameter = kernelResources->template GetParameter<PTX::PointerType<B, PTX::PointerType<B, T, PTX::GlobalSpace>>>(returnName);
			return addressGenerator.template GenerateAddress<T>(returnParameter, index);
		}
		else
		{
			Utils::Logger::LogError("Unable to generate address for return parameter " + NameUtils::ReturnName(returnIndex) " with shape " + Analysis::ShapeUtils::ShapeString(shape));
		}
	}
};

}
