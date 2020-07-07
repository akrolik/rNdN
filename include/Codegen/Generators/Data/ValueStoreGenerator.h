#pragma once

#include <string>

#include "Codegen/Generators/Generator.h"

#include "Analysis/Shape/Shape.h"
#include "Analysis/Shape/ShapeUtils.h"

#include "Codegen/Builder.h"
#include "Codegen/Generators/TypeDispatch.h"
#include "Codegen/Generators/Expressions/OperandGenerator.h"
#include "Codegen/Generators/Indexing/AddressGenerator.h"
#include "Codegen/Generators/Indexing/DataIndexGenerator.h"
#include "Codegen/Generators/Indexing/PrefixSumGenerator.h"
#include "Codegen/Generators/Indexing/ThreadGeometryGenerator.h"
#include "Codegen/Generators/Indexing/ThreadIndexGenerator.h"
#include "Codegen/Generators/Synchronization/AtomicGenerator.h"

#include "HorseIR/Tree/Tree.h"

#include "PTX/PTX.h"

namespace Codegen {

template<PTX::Bits B>
class ValueStoreGenerator : public Generator
{
public:
	using Generator::Generator;

	std::string Name() const override { return "ValueStoreGenerator"; }

	void Generate(const HorseIR::ReturnStatement *returnS)
	{
		auto returnIndex = 0u;
		for (const auto& operand : returnS->GetOperands())
		{
			DispatchType(*this, operand->GetType(), operand, returnIndex++);
		}
	}

	template<class T>
	void GenerateVector(const HorseIR::Operand *operand, unsigned int returnIndex)
	{
		m_cellIndex = 0;
		m_isCell = false;
		GenerateWrite<T>(operand, returnIndex);
	}

	template<class T>
	void GenerateList(const HorseIR::Operand *operand, unsigned int returnIndex)
	{
		auto& inputOptions = this->m_builder.GetInputOptions();
		if (inputOptions.IsVectorGeometry())
		{
			auto shape = inputOptions.ReturnWriteShapes.at(returnIndex);
			if (const auto listShape = Analysis::ShapeUtils::GetShape<Analysis::ListShape>(shape))
			{
				if (const auto size = Analysis::ShapeUtils::GetSize<Analysis::Shape::ConstantSize>(listShape->GetListSize()))
				{
					for (auto index = 0u; index < size->GetValue(); ++index)
					{
						m_cellIndex = index;
						m_isCell = true;
						GenerateWrite<T>(operand, returnIndex);
					}
				}
				else
				{
					Error("list-in-vector store for non-constant sized tuple " + Analysis::ShapeUtils::ShapeString(shape), returnIndex);
				}
			}
			else
			{
				Error("list-in-vector store for non-list shape " + Analysis::ShapeUtils::ShapeString(shape), returnIndex);
			}
		}
		else
		{
			// Store the list using a projection
			
			GenerateVector<T>(operand, returnIndex);
		}
	}

	template<class T>
	void GenerateTuple(unsigned int index, const HorseIR::Operand *operand, unsigned int returnIndex)
	{
		if (!this->m_builder.GetInputOptions().IsVectorGeometry())
		{
			Error("list store for heterogeneous tuple", returnIndex);
		}

		m_cellIndex = index;
		m_isCell = true;
		GenerateWrite<T>(operand, returnIndex);
	}

private:
	template<class T>
	void GenerateWrite(const HorseIR::Operand *operand, unsigned int returnIndex)
	{
		if constexpr(std::is_same<T, PTX::PredicateType>::value)
		{
			// Predicate (1-bit) values are stored as 8 bit integers on the CPU side so a conversion must first be run

			GenerateWrite<PTX::Int8Type>(operand, returnIndex);
		}
		else
		{
			auto& inputOptions = this->m_builder.GetInputOptions();

			// Select the write kind based on the thread geometry and return shape

			auto shape = inputOptions.ReturnWriteShapes.at(returnIndex);
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
						GenerateWriteReduction<T>(operand, OperandGenerator<B, T>::LoadKind::Vector, DataIndexGenerator<B>::Kind::Broadcast, returnIndex);
						return;
					}
					else if (*vectorGeometry == *vectorShape)
					{
						GenerateWriteVector<T>(operand, DataIndexGenerator<B>::Kind::VectorData, returnIndex);
						return;
					}
					else if (Analysis::ShapeUtils::IsSize<Analysis::Shape::CompressedSize>(vectorShape->GetSize()))
					{
						GenerateWriteCompressed<T>(operand, returnIndex);
						return;
					}
					else if (Analysis::ShapeUtils::IsDynamicShape(vectorShape))
					{
						// Otherwise, we expect the output to be handled separately
						return;
					}
				}
				else if (const auto listShape = Analysis::ShapeUtils::GetShape<Analysis::ListShape>(shape))
				{
					const auto cellShape = Analysis::ShapeUtils::MergeShapes(listShape->GetElementShapes());
					const auto cellVector = Analysis::ShapeUtils::GetShape<Analysis::VectorShape>(cellShape);

					// Only support vector writing for list-in-vector geometry

					if (*vectorGeometry == *cellVector)
					{
						GenerateWriteVector<T>(operand, DataIndexGenerator<B>::Kind::VectorData, returnIndex);
						return;
					}
					else if (Analysis::ShapeUtils::IsDynamicShape(cellVector))
					{
						// Otherwise, we expect the output to be handled separately
						return;
					}
				}
			}
			else if (const auto listGeometry = Analysis::ShapeUtils::GetShape<Analysis::ListShape>(inputOptions.ThreadGeometry))
			{
				if (const auto vectorShape = Analysis::ShapeUtils::GetShape<Analysis::VectorShape>(shape))
				{
					// Special horizontal write for @raze function

					const auto cellGeometry = Analysis::ShapeUtils::MergeShapes(listGeometry->GetElementShapes());
					if (const auto cellVectorGeometry = Analysis::ShapeUtils::GetShape<Analysis::VectorShape>(cellGeometry))
					{
						if (!Analysis::ShapeUtils::IsScalarSize(cellVectorGeometry->GetSize()))
						{
							GenerateWriteReduction<T>(operand, OperandGenerator<B, T>::LoadKind::ListData, DataIndexGenerator<B>::Kind::ListBroadcast, returnIndex);
							return;
						}
						else if (*listGeometry->GetListSize() == *vectorShape->GetSize())
						{
							GenerateWriteVector<T>(operand, DataIndexGenerator<B>::Kind::ListBroadcast, returnIndex);
							return;
						}
					}
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
							GenerateWriteReduction<T>(operand, OperandGenerator<B, T>::LoadKind::Vector, DataIndexGenerator<B>::Kind::Broadcast, returnIndex);
							return;
						}
						else if (*listShape == *listGeometry) // Compare the entire geometry, as the cells may differ
						{
							GenerateWriteVector<T>(operand, DataIndexGenerator<B>::Kind::ListData, returnIndex);
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
			
			Error("store for shape " + Analysis::ShapeUtils::ShapeString(shape), returnIndex);
		}
	}

	template<class T>
	void GenerateWriteReduction(const HorseIR::Operand *operand, typename OperandGenerator<B, T>::LoadKind loadKind, typename DataIndexGenerator<B>::Kind indexKind, unsigned int returnIndex)
	{
		auto resources = this->m_builder.GetLocalResources();

		// Get the reduction properties for the register and the write index

		auto value = GenerateWriteValue<T>(operand, loadKind);
		auto [granularity, op] = resources->GetReductionRegister(value);

		ThreadIndexGenerator<B> indexGenerator(this->m_builder);
		DataIndexGenerator<B> dataIndexGenerator(this->m_builder);
		auto writeIndex = dataIndexGenerator.GenerateIndex(indexKind);

		// Generate the reduction value depending on the reduced granularity

		switch (granularity)
		{
			case RegisterReductionGranularity::Single:
			{
				DataIndexGenerator<B> indexGenerator(this->m_builder);
				auto index = indexGenerator.GenerateDataIndex();
				GenerateWriteReduction(op, index, value, writeIndex, returnIndex);
				break;
			}
			case RegisterReductionGranularity::Warp:
			{
				auto laneid = indexGenerator.GenerateLaneIndex();
				GenerateWriteReduction(op, laneid, value, writeIndex, returnIndex);
				break;
			}
			case RegisterReductionGranularity::Block:
			{
				auto isVectorGeometry = this->m_builder.GetInputOptions().IsVectorGeometry();
				auto localIndex = (isVectorGeometry) ? indexGenerator.GenerateLocalIndex() : indexGenerator.GenerateListLocalIndex();
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
			case RegisterReductionOperation::None:
			{
				this->m_builder.AddStatement(new PTX::StoreInstruction<B, T, PTX::GlobalSpace>(address, value));
				break;
			}
			case RegisterReductionOperation::Add:
			{
				if constexpr(std::is_same<T, PTX::Int64Type>::value)
				{
					this->m_builder.AddStatement(new PTX::ReductionInstruction<B, PTX::UInt64Type, PTX::GlobalSpace, PTX::UInt64Type::ReductionOperation::Add>(
						new PTX::AddressAdapter<B, PTX::UInt64Type, PTX::Int64Type, PTX::GlobalSpace>(address),
						new PTX::Unsigned64RegisterAdapter(value)
					));
				}
				else
				{
					this->m_builder.AddStatement(new PTX::ReductionInstruction<B, T, PTX::GlobalSpace, T::ReductionOperation::Add>(address, value));
				}
				break;
			}
			case RegisterReductionOperation::Minimum:
			{
				// Float not supported by red
				if constexpr(PTX::is_float_type<T>::value)
				{
					AtomicGenerator<B> atomicGenerator(this->m_builder);
					atomicGenerator.GenerateMinMaxReduction(address, value, true);
				}
				else
				{
					this->m_builder.AddStatement(new PTX::ReductionInstruction<B, T, PTX::GlobalSpace, T::ReductionOperation::Minimum>(address, value));
				}
				break;
			}
			case RegisterReductionOperation::Maximum:
			{
				// Float not supported by red
				if constexpr(PTX::is_float_type<T>::value)
				{
					AtomicGenerator<B> atomicGenerator(this->m_builder);
					atomicGenerator.GenerateMinMaxReduction(address, value, false);
				}
				else
				{
					this->m_builder.AddStatement(new PTX::ReductionInstruction<B, T, PTX::GlobalSpace, T::ReductionOperation::Maximum>(address, value));
				}
				break;
			}
			default:
			{
				Error("store reduction operation", returnIndex);
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

		auto predicate = GenerateCASReductionOperation(reductionOp, global, value, returnIndex);

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
	const PTX::Register<PTX::PredicateType> *GenerateCASReductionOperation(RegisterReductionOperation reductionOp, const PTX::Register<T> *global, const PTX::Register<T> *value, unsigned int returnIndex)
	{
		if constexpr(std::is_same<T, PTX::Int8Type>::value)
		{
			auto convertedGlobal = ConversionGenerator::ConvertSource<PTX::Int16Type, T>(this->m_builder, global);
			auto convertedValue = ConversionGenerator::ConvertSource<PTX::Int16Type, T>(this->m_builder, value);

			auto predicate = GenerateCASReductionOperation(reductionOp, convertedGlobal, convertedValue, returnIndex);
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
					this->m_builder.AddStatement(new PTX::SetPredicateInstruction<T>(predicate, value, global, T::ComparisonOperator::Less));
					return predicate;
				}
				case RegisterReductionOperation::Maximum:
				{
					auto predicate = resources->template AllocateTemporary<PTX::PredicateType>();
					this->m_builder.AddStatement(new PTX::SetPredicateInstruction<T>(predicate, value, global, T::ComparisonOperator::Greater));
					return predicate;
				}
				default:
				{
					Error("store CAS reduction operation", returnIndex);
				}
			}
		}
	}

	template<class T>
	void GenerateWriteVector(const HorseIR::Operand *operand, typename DataIndexGenerator<B>::Kind indexKind, unsigned int returnIndex)
	{
		auto resources = this->m_builder.GetLocalResources();

		// Fetch the write address and store the value at the appropriate index (global indexing, or indexed register)

		auto value = GenerateWriteValue<T>(operand, OperandGenerator<B, T>::LoadKind::Vector);

		// Ensure the write is within bounds

		DataIndexGenerator<B> indexGenerator(this->m_builder);
		auto index = indexGenerator.GenerateIndex(indexKind);

		DataSizeGenerator<B> sizeGenerator(this->m_builder);
		auto size = sizeGenerator.GenerateSize(returnIndex, m_cellIndex);

		auto label = this->m_builder.CreateLabel("RET_" + std::to_string(returnIndex));
		auto sizePredicate = resources->template AllocateTemporary<PTX::PredicateType>();

		this->m_builder.AddStatement(new PTX::SetPredicateInstruction<PTX::UInt32Type>(sizePredicate, index, size, PTX::UInt32Type::ComparisonOperator::GreaterEqual));
		this->m_builder.AddStatement(new PTX::BranchInstruction(label, sizePredicate));

		// Store the value into global space

		auto writeIndex = resources->template GetIndexedRegister<T>(value);
		if (writeIndex == nullptr)
		{
			// Global/cell data indexing depending on thread geometry

			writeIndex = index;
		}

		auto address = GenerateAddress<T>(returnIndex, writeIndex);
		this->m_builder.AddStatement(new PTX::StoreInstruction<B, T, PTX::GlobalSpace>(address, value));

		this->m_builder.AddStatement(new PTX::BlankStatement());
		this->m_builder.AddStatement(label);
	}

	template<class T>
	void GenerateWriteCompressed(const HorseIR::Operand *operand, unsigned int returnIndex)
	{
		auto resources = this->m_builder.GetLocalResources();

		// Fetch the data and write to the compressed index

		auto value = GenerateWriteValue<T>(operand, OperandGenerator<B, T>::LoadKind::Vector);

		if (auto predicate = resources->template GetCompressedRegister<T>(value))
		{
			// Before generating compressed output, see if the write is indexed - if so, no compressed necessary

			auto writeIndex = resources->template GetIndexedRegister<T>(value);

			if (writeIndex == nullptr)
			{
				// Generate the in-order global prefix sum!

				auto kernelResources = this->m_builder.GetKernelResources();
				auto shape = this->m_builder.GetInputOptions().ReturnWriteShapes.at(returnIndex);
				if (Analysis::ShapeUtils::IsShape<Analysis::VectorShape>(shape))
				{
					auto parameter = kernelResources->GetParameter<PTX::PointerType<B, T>>(NameUtils::ReturnName(returnIndex));
					auto sizeParameter = kernelResources->GetParameter<PTX::PointerType<B, PTX::UInt32Type>>(NameUtils::SizeName(parameter));

					AddressGenerator<B, PTX::UInt32Type> addressGenerator(this->m_builder);
					auto sizeAddress = addressGenerator.GenerateAddress(sizeParameter);

					PrefixSumGenerator<B, PTX::UInt32Type> prefixSumGenerator(this->m_builder);
					writeIndex = prefixSumGenerator.template Generate<PTX::PredicateType>(sizeAddress, predicate, PrefixSumMode::Exclusive);
				}
				else if (Analysis::ShapeUtils::IsShape<Analysis::ListShape>(shape))
				{
					auto parameter = kernelResources->GetParameter<PTX::PointerType<B, PTX::PointerType<B, T, PTX::GlobalSpace>>>(NameUtils::ReturnName(returnIndex));
					auto sizeParameter = kernelResources->GetParameter<PTX::PointerType<B, PTX::PointerType<B, PTX::UInt32Type, PTX::GlobalSpace>>>(NameUtils::SizeName(parameter));

					//TODO: List prefix sum
					Error("list cell prefix sum", returnIndex);
				}
			}

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
			Error("compression predicate for return parameter", returnIndex);
		}
	}

	template<class T>
	PTX::Address<B, T, PTX::GlobalSpace> *GenerateAddress(unsigned int returnIndex, const PTX::TypedOperand<PTX::UInt32Type> *index)
	{
		// Get the address register for the return and offset by the index

		auto kernelResources = this->m_builder.GetKernelResources();
		auto returnName = NameUtils::ReturnName(returnIndex);

		auto shape = this->m_builder.GetInputOptions().ReturnWriteShapes.at(returnIndex);
		if (Analysis::ShapeUtils::IsShape<Analysis::VectorShape>(shape))
		{
			auto returnParameter = kernelResources->template GetParameter<PTX::PointerType<B, T>>(returnName);

			AddressGenerator<B, T> addressGenerator(this->m_builder);
			return addressGenerator.GenerateAddress(returnParameter, index);
		}
		else if (Analysis::ShapeUtils::IsShape<Analysis::ListShape>(shape))
		{
			auto returnParameter = kernelResources->template GetParameter<PTX::PointerType<B, PTX::PointerType<B, T, PTX::GlobalSpace>>>(returnName);

			AddressGenerator<B, T> addressGenerator(this->m_builder);
			if (m_isCell)
			{
				return addressGenerator.GenerateAddress(returnParameter, m_cellIndex, index);
			}
			return addressGenerator.GenerateAddress(returnParameter, index);
		}
		Error("address for return parameter with shape " + Analysis::ShapeUtils::ShapeString(shape), returnIndex);
	}

	template<class T>
	const PTX::Register<T> *GenerateWriteValue(const HorseIR::Operand *operand, typename OperandGenerator<B, T>::LoadKind loadKind)
	{
		OperandGenerator<B, T> operandGenerator(this->m_builder);
		if (m_isCell)
		{
			return operandGenerator.GenerateRegister(operand, loadKind, m_cellIndex);
		}
		return operandGenerator.GenerateRegister(operand, loadKind);
	}

	[[noreturn]] void Error(const std::string& message, unsigned int returnIndex) const
	{
		Generator::Error(message + " [index = " + std::to_string(returnIndex) + "]");
	}

	unsigned int m_cellIndex = 0;
	bool m_isCell = false;
};

}
