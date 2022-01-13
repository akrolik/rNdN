#pragma once

#include <string>

#include "Frontend/Codegen/Generators/Generator.h"

#include "Frontend/Codegen/Builder.h"
#include "Frontend/Codegen/Generators/TypeDispatch.h"
#include "Frontend/Codegen/Generators/Expressions/OperandGenerator.h"
#include "Frontend/Codegen/Generators/Indexing/AddressGenerator.h"
#include "Frontend/Codegen/Generators/Indexing/DataIndexGenerator.h"
#include "Frontend/Codegen/Generators/Indexing/PrefixSumGenerator.h"
#include "Frontend/Codegen/Generators/Indexing/ThreadGeometryGenerator.h"
#include "Frontend/Codegen/Generators/Indexing/ThreadIndexGenerator.h"
#include "Frontend/Codegen/Generators/Synchronization/AtomicGenerator.h"

#include "HorseIR/Analysis/Shape/Shape.h"
#include "HorseIR/Analysis/Shape/ShapeUtils.h"

#include "HorseIR/Tree/Tree.h"

#include "PTX/Tree/Tree.h"

namespace Frontend {
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
			if (const auto listShape = HorseIR::Analysis::ShapeUtils::GetShape<HorseIR::Analysis::ListShape>(shape))
			{
				if (const auto size = HorseIR::Analysis::ShapeUtils::GetSize<HorseIR::Analysis::Shape::ConstantSize>(listShape->GetListSize()))
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
					Error("list-in-vector store for non-constant sized tuple " + HorseIR::Analysis::ShapeUtils::ShapeString(shape), returnIndex);
				}
			}
			else
			{
				Error("list-in-vector store for non-list shape " + HorseIR::Analysis::ShapeUtils::ShapeString(shape), returnIndex);
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
			if (const auto vectorGeometry = HorseIR::Analysis::ShapeUtils::GetShape<HorseIR::Analysis::VectorShape>(inputOptions.ThreadGeometry))
			{
				if (const auto vectorShape = HorseIR::Analysis::ShapeUtils::GetShape<HorseIR::Analysis::VectorShape>(shape))
				{
					// Check for the style of write:
					//  (1) Reduction (we assume this corresponds to scalar output in a non-scalar kernel)
					//  (2) Vector
					//  (3) Compression

					if (HorseIR::Analysis::ShapeUtils::IsScalarSize(vectorShape->GetSize()) && !HorseIR::Analysis::ShapeUtils::IsScalarSize(vectorGeometry->GetSize()))
					{
						GenerateWriteReduction<T>(operand, OperandGenerator<B, T>::LoadKind::Vector, DataIndexGenerator<B>::Kind::Broadcast, returnIndex);
						return;
					}
					else if (*vectorGeometry == *vectorShape)
					{
						GenerateWriteVector<T>(operand, DataIndexGenerator<B>::Kind::VectorData, returnIndex);
						return;
					}
					else if (HorseIR::Analysis::ShapeUtils::IsSize<HorseIR::Analysis::Shape::CompressedSize>(vectorShape->GetSize()))
					{
						GenerateWriteCompressed<T>(operand, returnIndex);
						return;
					}
					else if (HorseIR::Analysis::ShapeUtils::IsDynamicShape(vectorShape))
					{
						// Otherwise, we expect the output to be handled separately
						return;
					}
				}
				else if (const auto listShape = HorseIR::Analysis::ShapeUtils::GetShape<HorseIR::Analysis::ListShape>(shape))
				{
					const auto cellShape = HorseIR::Analysis::ShapeUtils::MergeShapes(listShape->GetElementShapes());
					const auto cellVector = HorseIR::Analysis::ShapeUtils::GetShape<HorseIR::Analysis::VectorShape>(cellShape);

					// Only support vector writing for list-in-vector geometry

					if (*vectorGeometry == *cellVector)
					{
						GenerateWriteVector<T>(operand, DataIndexGenerator<B>::Kind::VectorData, returnIndex);
						return;
					}
					else if (HorseIR::Analysis::ShapeUtils::IsDynamicShape(cellVector))
					{
						// Otherwise, we expect the output to be handled separately
						return;
					}
				}
			}
			else if (const auto listGeometry = HorseIR::Analysis::ShapeUtils::GetShape<HorseIR::Analysis::ListShape>(inputOptions.ThreadGeometry))
			{
				if (const auto vectorShape = HorseIR::Analysis::ShapeUtils::GetShape<HorseIR::Analysis::VectorShape>(shape))
				{
					// Special horizontal write for @raze function

					const auto cellGeometry = HorseIR::Analysis::ShapeUtils::MergeShapes(listGeometry->GetElementShapes());
					if (const auto cellVectorGeometry = HorseIR::Analysis::ShapeUtils::GetShape<HorseIR::Analysis::VectorShape>(cellGeometry))
					{
						if (!HorseIR::Analysis::ShapeUtils::IsScalarSize(cellVectorGeometry->GetSize()))
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
				else if (const auto listShape = HorseIR::Analysis::ShapeUtils::GetShape<HorseIR::Analysis::ListShape>(shape))
				{
					// Check for the style of write:
					//  (1) Reduction (we assume this corresponds to scalar output in a non-scalar cell)
					//  (2) List

					const auto cellShape = HorseIR::Analysis::ShapeUtils::MergeShapes(listShape->GetElementShapes());
					const auto cellGeometry = HorseIR::Analysis::ShapeUtils::MergeShapes(listGeometry->GetElementShapes());

					const auto cellVector = HorseIR::Analysis::ShapeUtils::GetShape<HorseIR::Analysis::VectorShape>(cellShape);
					const auto cellVectorGeometry = HorseIR::Analysis::ShapeUtils::GetShape<HorseIR::Analysis::VectorShape>(cellGeometry);

					if (cellVector && cellVectorGeometry)
					{
						if (HorseIR::Analysis::ShapeUtils::IsScalarSize(cellVector->GetSize()) && !HorseIR::Analysis::ShapeUtils::IsScalarSize(cellVectorGeometry->GetSize()))
						{
							GenerateWriteReduction<T>(operand, OperandGenerator<B, T>::LoadKind::Vector, DataIndexGenerator<B>::Kind::Broadcast, returnIndex);
							return;
						}
						else if (*listShape == *listGeometry) // Compare the entire geometry, as the cells may differ
						{
							GenerateWriteVector<T>(operand, DataIndexGenerator<B>::Kind::ListData, returnIndex);
							return;
						}
						else if (HorseIR::Analysis::ShapeUtils::IsSize<HorseIR::Analysis::Shape::CompressedSize>(cellVector->GetSize()))
						{
							GenerateWriteCompressed<T>(operand, returnIndex);
							return;
						}
					}
				}
			}
			
			Error("store for shape " + HorseIR::Analysis::ShapeUtils::ShapeString(shape), returnIndex);
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
	void GenerateWriteReduction(RegisterReductionOperation reductionOp, PTX::TypedOperand<PTX::UInt32Type> *activeIndex, PTX::Register<T> *value, PTX::TypedOperand<PTX::UInt32Type> *writeIndex, unsigned int returnIndex)
	{
		auto resources = this->m_builder.GetLocalResources();

		// At the end of the partial reduction we only have a single active thread. Use it to store the final value

		this->m_builder.AddIfStatement("RET_" + std::to_string(returnIndex), [&]()
		{
			auto predicate = resources->template AllocateTemporary<PTX::PredicateType>();
			this->m_builder.AddStatement(new PTX::SetPredicateInstruction<PTX::UInt32Type>(
				predicate, activeIndex, new PTX::UInt32Value(0), PTX::UInt32Type::ComparisonOperator::NotEqual
			));
			return std::make_tuple(predicate, false);
		},
		[&]()
		{
			// Get the address of the write location depending on the indexing kind and the reduction operation

			auto address = GenerateAddress<T>(returnIndex, writeIndex);
			GenerateWriteReduction(reductionOp, address, value, returnIndex);
		});
	}

	template<class T>
	void GenerateWriteReduction(RegisterReductionOperation reductionOp, PTX::Address<B, T, PTX::GlobalSpace> *address, PTX::Register<T> *value, unsigned int returnIndex)
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
	void GenerateReductionInstruction(RegisterReductionOperation reductionOp, PTX::Address<B, T, PTX::GlobalSpace> *address, PTX::Register<T> *value, unsigned int returnIndex)
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
					this->m_builder.AddStatement(new PTX::ReductionInstruction<B, PTX::UInt64Type, PTX::GlobalSpace>(
						new PTX::AddressAdapter<B, PTX::UInt64Type, PTX::Int64Type, PTX::GlobalSpace>(address),
						new PTX::Unsigned64RegisterAdapter(value),
						PTX::UInt64Type::ReductionOperation::Add
					));
				}
				else
				{
					this->m_builder.AddStatement(new PTX::ReductionInstruction<B, T, PTX::GlobalSpace>(
						address, value, T::ReductionOperation::Add
					));
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
					this->m_builder.AddStatement(new PTX::ReductionInstruction<B, T, PTX::GlobalSpace>(
						address, value, T::ReductionOperation::Minimum
					));
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
					this->m_builder.AddStatement(new PTX::ReductionInstruction<B, T, PTX::GlobalSpace>(
						address, value, T::ReductionOperation::Maximum
					));
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
	void GenerateCASWriteReduction(RegisterReductionOperation reductionOp, PTX::Address<B, T, PTX::GlobalSpace> *address, PTX::Register<T> *value, unsigned int returnIndex)
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

		// Generate the reduction (CUDA programming guide)

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
	PTX::Register<PTX::PredicateType> *GenerateCASReductionOperation(RegisterReductionOperation reductionOp, PTX::Register<T> *global, PTX::Register<T> *value, unsigned int returnIndex)
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

		auto writeIndex = resources->template GetIndexedRegister<T>(value);
		if (writeIndex == nullptr)
		{
			// Global/cell data indexing depending on thread geometry

			DataIndexGenerator<B> indexGenerator(this->m_builder);
			writeIndex = indexGenerator.GenerateIndex(indexKind);
		}

		// Ensure the write is within bounds

		this->m_builder.AddIfStatement("RET_DATA_" + std::to_string(returnIndex), [&]()
		{
			DataSizeGenerator<B> sizeGenerator(this->m_builder);
			auto size = sizeGenerator.GenerateSize(returnIndex, m_cellIndex);

			auto predicate = resources->template AllocateTemporary<PTX::PredicateType>();
			this->m_builder.AddStatement(new PTX::SetPredicateInstruction<PTX::UInt32Type>(
				predicate, writeIndex, size, PTX::UInt32Type::ComparisonOperator::GreaterEqual
			));

			return std::make_tuple(predicate, false);
		},
		[&]()
		{
			this->m_builder.AddIfStatement("RET_GEO_" + std::to_string(returnIndex), [&]()
			{
				DataIndexGenerator<B> indexGenerator(this->m_builder);
				auto index = indexGenerator.GenerateDataIndex();

				ThreadGeometryGenerator<B> geometryGenerator(this->m_builder);
				auto geometry = geometryGenerator.GenerateDataGeometry();

				auto predicate = resources->template AllocateTemporary<PTX::PredicateType>();
				this->m_builder.AddStatement(new PTX::SetPredicateInstruction<PTX::UInt32Type>(
					predicate, index, geometry, PTX::UInt32Type::ComparisonOperator::GreaterEqual
				));

				return std::make_tuple(predicate, false);
			},
			[&]()
			{
				// Store the value into global space

				auto address = GenerateAddress<T>(returnIndex, writeIndex);
				this->m_builder.AddStatement(new PTX::StoreInstruction<B, T, PTX::GlobalSpace>(address, value));
			});
		});
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
				if (HorseIR::Analysis::ShapeUtils::IsShape<HorseIR::Analysis::VectorShape>(shape))
				{
					auto parameter = kernelResources->GetParameter<PTX::PointerType<B, T>>(NameUtils::ReturnName(returnIndex));
					auto sizeParameter = kernelResources->GetParameter<PTX::PointerType<B, PTX::UInt32Type>>(NameUtils::SizeName(parameter));

					AddressGenerator<B, PTX::UInt32Type> addressGenerator(this->m_builder);
					auto sizeAddress = addressGenerator.GenerateAddress(sizeParameter);

					if (m_prefixSums.find(predicate) != m_prefixSums.end())
					{
						writeIndex = m_prefixSums.at(predicate);

						this->m_builder.AddIfStatement("COPY_" + std::to_string(returnIndex), [&]()
						{
							auto sizePredicate = resources->template AllocateTemporary<PTX::PredicateType>();
							auto lastIndex = resources->template AllocateTemporary<PTX::UInt32Type>();

							ThreadIndexGenerator<B> indexGenerator(this->m_builder);
							auto index = indexGenerator.GenerateGlobalIndex();

							ThreadGeometryGenerator<B> geometryGenerator(this->m_builder);
							auto size = geometryGenerator.GenerateGlobalSize();

							this->m_builder.AddStatement(new PTX::SubtractInstruction<PTX::UInt32Type>(lastIndex, size, new PTX::UInt32Value(1)));
							this->m_builder.AddStatement(new PTX::SetPredicateInstruction<PTX::UInt32Type>(
								sizePredicate, index, lastIndex, PTX::UInt32Type::ComparisonOperator::Equal
							));

							return std::make_pair(sizePredicate, true);
						},
						[&]()
						{
							auto size = m_prefixSizes.at(predicate);
							this->m_builder.AddStatement(new PTX::StoreInstruction<B, PTX::UInt32Type, PTX::GlobalSpace>(sizeAddress, size));
						});
					}
					else
					{
						PrefixSumGenerator<B, PTX::UInt32Type> prefixSumGenerator(this->m_builder);
						auto [inclusive, exclusive] = prefixSumGenerator.template Generate<PTX::PredicateType>(sizeAddress, predicate);

						writeIndex = exclusive;

						m_prefixSizes[predicate] = inclusive;
						m_prefixSums[predicate] = writeIndex;
					}
				}
				else if (HorseIR::Analysis::ShapeUtils::IsShape<HorseIR::Analysis::ListShape>(shape))
				{
					auto parameter = kernelResources->GetParameter<PTX::PointerType<B, PTX::PointerType<B, T, PTX::GlobalSpace>>>(NameUtils::ReturnName(returnIndex));
					auto sizeParameter = kernelResources->GetParameter<PTX::PointerType<B, PTX::PointerType<B, PTX::UInt32Type, PTX::GlobalSpace>>>(NameUtils::SizeName(parameter));

					//TODO: List prefix sum
					Error("list cell prefix sum", returnIndex);
				}
			}

			// Check for compression - this will mask outputs

			this->m_builder.AddIfStatement("RET_" + std::to_string(returnIndex), [&]()
			{
				return std::make_tuple(predicate, true);
			},
			[&]()
			{
				// Store the value at the place specified by the prefix sum

				auto address = GenerateAddress<T>(returnIndex, writeIndex);
				this->m_builder.AddStatement(new PTX::StoreInstruction<B, T, PTX::GlobalSpace>(address, value));
			});
		}
		else
		{
			Error("compression predicate for return parameter", returnIndex);
		}
	}

	template<class T>
	PTX::Address<B, T, PTX::GlobalSpace> *GenerateAddress(unsigned int returnIndex, PTX::TypedOperand<PTX::UInt32Type> *index)
	{
		// Get the address register for the return and offset by the index

		auto kernelResources = this->m_builder.GetKernelResources();
		auto returnName = NameUtils::ReturnName(returnIndex);

		auto shape = this->m_builder.GetInputOptions().ReturnWriteShapes.at(returnIndex);
		if (HorseIR::Analysis::ShapeUtils::IsShape<HorseIR::Analysis::VectorShape>(shape))
		{
			auto returnParameter = kernelResources->template GetParameter<PTX::PointerType<B, T>>(returnName);

			AddressGenerator<B, T> addressGenerator(this->m_builder);
			return addressGenerator.GenerateAddress(returnParameter, index);
		}
		else if (HorseIR::Analysis::ShapeUtils::IsShape<HorseIR::Analysis::ListShape>(shape))
		{
			auto returnParameter = kernelResources->template GetParameter<PTX::PointerType<B, PTX::PointerType<B, T, PTX::GlobalSpace>>>(returnName);

			AddressGenerator<B, T> addressGenerator(this->m_builder);
			if (m_isCell)
			{
				return addressGenerator.GenerateAddress(returnParameter, m_cellIndex, index);
			}
			return addressGenerator.GenerateAddress(returnParameter, index);
		}
		Error("address for return parameter with shape " + HorseIR::Analysis::ShapeUtils::ShapeString(shape), returnIndex);
	}

	template<class T>
	PTX::Register<T> *GenerateWriteValue(const HorseIR::Operand *operand, typename OperandGenerator<B, T>::LoadKind loadKind)
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

	robin_hood::unordered_map<PTX::Register<PTX::PredicateType> *, PTX::TypedOperand<PTX::UInt32Type> *> m_prefixSums;
	robin_hood::unordered_map<PTX::Register<PTX::PredicateType> *, PTX::Register<PTX::UInt32Type> *> m_prefixSizes;
};

}
}
