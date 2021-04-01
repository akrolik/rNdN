#pragma once

#include "Frontend/Codegen/Generators/Generator.h"
#include "HorseIR/Traversal/ConstVisitor.h"

#include "Frontend/Codegen/Builder.h"
#include "Frontend/Codegen/Generators/Data/ParameterGenerator.h"
#include "Frontend/Codegen/Generators/Data/ValueLoadGenerator.h"
#include "Frontend/Codegen/Generators/Expressions/OperandGenerator.h"
#include "Frontend/Codegen/Generators/Expressions/Builtins/InternalCacheGenerator.h"
#include "Frontend/Codegen/Generators/Indexing/AddressGenerator.h"
#include "Frontend/Codegen/Generators/Indexing/DataIndexGenerator.h"
#include "Frontend/Codegen/Generators/Indexing/ThreadIndexGenerator.h"

#include "HorseIR/Analysis/Shape/Shape.h"
#include "HorseIR/Analysis/Shape/ShapeUtils.h"

#include "HorseIR/Tree/Tree.h"
#include "HorseIR/Utils/LiteralUtils.h"

#include "PTX/Tree/Tree.h"

namespace Frontend {
namespace Codegen {

enum class OrderMode {
	Shared,
	Global
};

template<PTX::Bits B, unsigned int SORT_CACHE_SIZE>
class OrderLoadGenerator : public Generator, public HorseIR::ConstVisitor
{
public:
	OrderLoadGenerator(Builder& builder, OrderMode mode) : Generator(builder), m_mode(mode) {}

	std::string Name() const override { return "OrderLoadGenerator"; }

	void Generate(const HorseIR::Operand *dataArgument, PTX::Register<PTX::UInt32Type> *leftIndex, PTX::Register<PTX::UInt32Type> *rightIndex)
	{
		m_leftIndex = leftIndex;
		m_rightIndex = rightIndex;
		dataArgument->Accept(*this);
	}

	void Visit(const HorseIR::Identifier *identifier) override
	{
		DispatchType(*this, identifier->GetType(), identifier);
	}

	void Visit(const HorseIR::Literal *literal) override
	{
		Error("literal data for @order");
	}

	template<class T>
	void GenerateVector(const HorseIR::Identifier *identifier)
	{
		GenerateLoad<T>(identifier);
	}

	template<class T>
	void GenerateList(const HorseIR::Identifier *identifier)
	{
		const auto& inputOptions = this->m_builder.GetInputOptions();
		const auto parameter = inputOptions.Parameters.at(identifier->GetSymbol());
		const auto shape = inputOptions.ParameterShapes.at(parameter);

		if (const auto listShape = HorseIR::Analysis::ShapeUtils::GetShape<HorseIR::Analysis::ListShape>(shape))
		{
			if (const auto size = HorseIR::Analysis::ShapeUtils::GetSize<HorseIR::Analysis::Shape::ConstantSize>(listShape->GetListSize()))
			{
				for (auto index = 0u; index < size->GetValue(); ++index)
				{
					GenerateTuple<T>(index, identifier);
				}
				return;
			}
		}
		Error("non-constant cell count");
	}

	template<class T>
	void GenerateTuple(unsigned int index, const HorseIR::Identifier *identifier)
	{
		GenerateLoad<T>(identifier, true, index);
	}

private:
	template<class T>
	void GenerateLoad(const HorseIR::Identifier *identifier, bool isCell = false, unsigned int cellIndex = 0)
	{
		if constexpr(std::is_same<T, PTX::PredicateType>::value)
		{
			GenerateLoad<PTX::Int8Type>(identifier, isCell, cellIndex);
		}
		else
		{
			switch (m_mode)
			{
				case OrderMode::Shared:
				{
					// Get the cache addresses of left/right values
					
					auto kernelResources = this->m_builder.GetKernelResources();

					auto cacheName = identifier->GetName() + "_cache" + ((isCell) ? std::to_string(cellIndex) : "");
					auto s_cache = new PTX::ArrayVariableAdapter<T, SORT_CACHE_SIZE * 2, PTX::SharedSpace>(
						kernelResources->template GetSharedVariable<PTX::ArrayType<T, SORT_CACHE_SIZE * 2>>(cacheName)
					);

					AddressGenerator<B, T> addressGenerator(this->m_builder);
					auto s_leftAddress = addressGenerator.GenerateAddress(s_cache, m_leftIndex);
					auto s_rightAddress = addressGenerator.GenerateAddress(s_cache, m_rightIndex);

					// Load the values

					auto leftName = NameUtils::VariableName(identifier, isCell, cellIndex, "left");
					auto rightName = NameUtils::VariableName(identifier, isCell, cellIndex, "right");

					auto resources = this->m_builder.GetLocalResources();
					auto leftValue = resources->template AllocateRegister<T>(leftName);
					auto rightValue = resources->template AllocateRegister<T>(rightName);

					this->m_builder.AddStatement(new PTX::LoadInstruction<B, T, PTX::SharedSpace>(leftValue, s_leftAddress));
					this->m_builder.AddStatement(new PTX::LoadInstruction<B, T, PTX::SharedSpace>(rightValue, s_rightAddress));

					break;
				}
				case OrderMode::Global:
				{
					OperandGenerator<B, T> operandGenerator(this->m_builder);
					operandGenerator.SetBoundsCheck(false);
					operandGenerator.GenerateOperand(identifier, m_leftIndex, "left", cellIndex);
					operandGenerator.GenerateOperand(identifier, m_rightIndex, "right", cellIndex);

					break;
				}
			}
		}
	}
	
	OrderMode m_mode = OrderMode::Shared;

	PTX::Register<PTX::UInt32Type> *m_leftIndex = nullptr;
	PTX::Register<PTX::UInt32Type> *m_rightIndex = nullptr;
};

template<PTX::Bits B>
class OrderComparisonGenerator : public Generator, public HorseIR::ConstVisitor
{
public:
	enum class Order {
		Ascending,
		Descending
	};

	OrderComparisonGenerator(Builder& builder, Order sequenceOrder, PTX::Register<PTX::PredicateType> *swapPredicate)
		: Generator(builder), m_sequenceOrder(sequenceOrder), m_swapPredicate(swapPredicate) {}

	std::string Name() const override { return "OrderComparisonGenerator"; }

	void Generate(const HorseIR::Operand *dataArgument, const HorseIR::TypedVectorLiteral<std::int8_t> *orderLiteral)
	{
		auto resources = this->m_builder.GetLocalResources();
		m_endPredicate = resources->template AllocateTemporary<PTX::PredicateType>();
		this->m_builder.AddStatement(new PTX::MoveInstruction<PTX::PredicateType>(m_endPredicate, new PTX::BoolValue(false)));

		m_orderLiteral = orderLiteral;
		dataArgument->Accept(*this);
	}

	void Visit(const HorseIR::Identifier *identifier) override
	{
		DispatchType(*this, identifier->GetType(), identifier);
	}

	void Visit(const HorseIR::Literal *literal) override
	{
		Error("literal data for @order");
	}

	template<class T>
	void GenerateVector(const HorseIR::Identifier *identifier)
	{
		GenerateComparison<T>(identifier);
	}

	template<class T>
	void GenerateList(const HorseIR::Identifier *identifier)
	{
		const auto& inputOptions = this->m_builder.GetInputOptions();
		const auto parameter = inputOptions.Parameters.at(identifier->GetSymbol());
		const auto shape = inputOptions.ParameterShapes.at(parameter);

		if (const auto listShape = HorseIR::Analysis::ShapeUtils::GetShape<HorseIR::Analysis::ListShape>(shape))
		{
			if (const auto size = HorseIR::Analysis::ShapeUtils::GetSize<HorseIR::Analysis::Shape::ConstantSize>(listShape->GetListSize()))
			{
				for (auto index = 0u; index < size->GetValue(); ++index)
				{
					GenerateComparison<T>(identifier, true, index, size->GetValue());
				}
				return;
			}
		}
		Error("non-constant cell count");
	}

	template<class T>
	void GenerateTuple(unsigned int index, const HorseIR::Identifier *identifier)
	{
		const auto& inputOptions = this->m_builder.GetInputOptions();
		const auto parameter = inputOptions.Parameters.at(identifier->GetSymbol());
		const auto shape = inputOptions.ParameterShapes.at(parameter);

		if (const auto listShape = HorseIR::Analysis::ShapeUtils::GetShape<HorseIR::Analysis::ListShape>(shape))
		{
			if (const auto size = HorseIR::Analysis::ShapeUtils::GetSize<HorseIR::Analysis::Shape::ConstantSize>(listShape->GetListSize()))
			{
				GenerateComparison<T>(identifier, true, index, size->GetValue());
				return;
			}
		}
		Error("non-constant cell count");
	}

private:
	template<class T>
	void GenerateComparison(const HorseIR::Identifier *identifier, bool isCell = false, unsigned int index = 0, unsigned int limit = 1)
	{
		if constexpr(std::is_same<T, PTX::PredicateType>::value || std::is_same<T, PTX::Int8Type>::value)
		{
			GenerateComparison<PTX::Int16Type>(identifier, isCell, index, limit);
		}
		else
		{
			this->m_builder.AddIfStatement("COMP", [&]()
			{
				return std::make_tuple(m_endPredicate, false);
			},
			[&]()
			{
				OperandGenerator<B, T> operandGenerator(this->m_builder);
				auto leftValue = operandGenerator.GenerateOperand(identifier, nullptr, "left", index);
				auto rightValue = operandGenerator.GenerateOperand(identifier, nullptr, "right", index);

				auto dataOrder = (m_orderLiteral->GetValue((m_orderLiteral->GetCount() == 1 ? 0 : index))) ? Order::Ascending : Order::Descending;
				auto comparison = (m_sequenceOrder == dataOrder) ? T::ComparisonOperator::Less : T::ComparisonOperator::Greater;

				this->m_builder.AddStatement(new PTX::SetPredicateInstruction<T>(m_swapPredicate, leftValue, rightValue, comparison));

				// Branch if the predicate is true

				if (index + 1 < limit)
				{
					// Check for the next branch

					auto resources = this->m_builder.GetLocalResources();
					auto nequalPredicate = resources->template AllocateTemporary<PTX::PredicateType>();

					this->m_builder.AddStatement(new PTX::SetPredicateInstruction<T>(
						nequalPredicate, leftValue, rightValue, T::ComparisonOperator::NotEqual
					));
					this->m_builder.AddStatement(new PTX::OrInstruction<PTX::PredicateType>(
						m_endPredicate, m_swapPredicate, nequalPredicate
					));

				}
			});
		}
	}

private:
	const HorseIR::TypedVectorLiteral<std::int8_t> *m_orderLiteral = nullptr;
	Order m_sequenceOrder;

	PTX::Register<PTX::PredicateType> *m_swapPredicate = nullptr;
	PTX::Register<PTX::PredicateType> *m_endPredicate = nullptr;
};

template<PTX::Bits B, unsigned int SORT_CACHE_SIZE>
class OrderSwapGenerator : public Generator, public HorseIR::ConstVisitor
{
public:
	OrderSwapGenerator(Builder& builder, OrderMode mode) : Generator(builder), m_mode(mode) {}

	std::string Name() const override { return "OrderSwapGenerator"; }

	void Generate(const HorseIR::Operand *index, const HorseIR::Operand *dataArgument, PTX::Register<PTX::UInt32Type> *leftIndex, PTX::Register<PTX::UInt32Type> *rightIndex)
	{
		m_leftIndex = leftIndex;
		m_rightIndex = rightIndex;
		index->Accept(*this);
		dataArgument->Accept(*this);
	}

	void Visit(const HorseIR::Identifier *identifier) override
	{
		DispatchType(*this, identifier->GetType(), identifier);
	}

	void Visit(const HorseIR::Literal *literal) override
	{
		Error("literal data for @order");
	}

	template<class T>
	void GenerateVector(const HorseIR::Identifier *identifier)
	{
		switch (m_mode)
		{
			case OrderMode::Shared:
			{
				GenerateSharedSwap<T>(identifier);
				break;
			}
			case OrderMode::Global:
			{
				GenerateGlobalSwap<T>(identifier);
				break;
			}
		}
	}

	template<class T>
	void GenerateList(const HorseIR::Identifier *identifier)
	{
		const auto& inputOptions = this->m_builder.GetInputOptions();
		const auto parameter = inputOptions.Parameters.at(identifier->GetSymbol());
		const auto shape = inputOptions.ParameterShapes.at(parameter);

		if (const auto listShape = HorseIR::Analysis::ShapeUtils::GetShape<HorseIR::Analysis::ListShape>(shape))
		{
			if (const auto size = HorseIR::Analysis::ShapeUtils::GetSize<HorseIR::Analysis::Shape::ConstantSize>(listShape->GetListSize()))
			{
				for (auto index = 0u; index < size->GetValue(); ++index)
				{
					GenerateTuple<T>(index, identifier);
				}
				return;
			}
		}
		Error("non-constant cell count");
	}

	template<class T>
	void GenerateTuple(unsigned int index, const HorseIR::Identifier *identifier)
	{
		switch (m_mode)
		{
			case OrderMode::Shared:
			{
				GenerateSharedSwap<T>(identifier, true, index);
				break;
			}
			case OrderMode::Global:
			{
				GenerateGlobalSwap<T>(identifier, true, index);
				break;
			}
		}
	}

private:
	template<class T>
	void GenerateSharedSwap(const HorseIR::Identifier *identifier, bool isCell = false, unsigned int cellIndex = 0)
	{
		if constexpr(std::is_same<T, PTX::PredicateType>::value)
		{
			GenerateSharedSwap<PTX::Int8Type>(identifier, isCell, cellIndex);
		}
		else
		{
			// Get the cache addresses of left/right values
			
			auto kernelResources = this->m_builder.GetKernelResources();

			auto cacheName = identifier->GetName() + "_cache" + ((isCell) ? std::to_string(cellIndex) : "");
			auto s_cache = new PTX::ArrayVariableAdapter<T, SORT_CACHE_SIZE * 2, PTX::SharedSpace>(
				kernelResources->template GetSharedVariable<PTX::ArrayType<T, SORT_CACHE_SIZE * 2>>(cacheName)
			);

			AddressGenerator<B, T> addressGenerator(this->m_builder);
			auto s_leftAddress = addressGenerator.GenerateAddress(s_cache, m_leftIndex);
			auto s_rightAddress = addressGenerator.GenerateAddress(s_cache, m_rightIndex);

			// Load the values

			auto leftName = NameUtils::VariableName(identifier, isCell, cellIndex, "left");
			auto rightName = NameUtils::VariableName(identifier, isCell, cellIndex, "right");

			auto resources = this->m_builder.GetLocalResources();
			auto leftValue = resources->template GetRegister<T>(leftName);
			auto rightValue = resources->template GetRegister<T>(rightName);

			this->m_builder.AddStatement(new PTX::StoreInstruction<B, T, PTX::SharedSpace>(s_leftAddress, rightValue));
			this->m_builder.AddStatement(new PTX::StoreInstruction<B, T, PTX::SharedSpace>(s_rightAddress, leftValue));
		}
	}

	template<class T>
	void GenerateGlobalSwap(const HorseIR::Identifier *identifier, bool isCell = false, unsigned int cellIndex = 0)
	{
		if constexpr(std::is_same<T, PTX::PredicateType>::value)
		{
			GenerateGlobalSwap<PTX::Int8Type>(identifier, isCell, cellIndex);
		}
		else
		{
			// Swap the left and right values in global memory

			auto resources = this->m_builder.GetLocalResources();
			auto kernelResources = this->m_builder.GetKernelResources();
			
			// Get the left and right values

			OperandGenerator<B, T> operandGenerator(this->m_builder);
			auto leftValue = operandGenerator.GenerateRegister(identifier, m_leftIndex, "left", cellIndex);
			auto rightValue = operandGenerator.GenerateRegister(identifier, m_rightIndex, "right", cellIndex);

			// Get the addresses of the left and right positions

			if (isCell)
			{
				auto parameter = kernelResources->template GetParameter<PTX::PointerType<B, PTX::PointerType<B, T, PTX::GlobalSpace>>>(NameUtils::VariableName(identifier));

				AddressGenerator<B, T> addressGenerator(this->m_builder);
				auto leftAddress = addressGenerator.GenerateAddress(parameter, cellIndex, m_leftIndex);
				auto rightAddress = addressGenerator.GenerateAddress(parameter, cellIndex, m_rightIndex);

				// Store the results back

				this->m_builder.AddStatement(new PTX::StoreInstruction<B, T, PTX::GlobalSpace>(leftAddress, rightValue));
				this->m_builder.AddStatement(new PTX::StoreInstruction<B, T, PTX::GlobalSpace>(rightAddress, leftValue));
			}
			else
			{
				auto parameter = kernelResources->template GetParameter<PTX::PointerType<B, T>>(NameUtils::VariableName(identifier));

				AddressGenerator<B, T> addressGenerator(this->m_builder);
				auto leftAddress = addressGenerator.GenerateAddress(parameter, m_leftIndex);
				auto rightAddress = addressGenerator.GenerateAddress(parameter, m_rightIndex);

				// Store the results back

				this->m_builder.AddStatement(new PTX::StoreInstruction<B, T, PTX::GlobalSpace>(leftAddress, rightValue));
				this->m_builder.AddStatement(new PTX::StoreInstruction<B, T, PTX::GlobalSpace>(rightAddress, leftValue));
			}
		}
	}

	OrderMode m_mode = OrderMode::Shared;

	PTX::Register<PTX::UInt32Type> *m_leftIndex = nullptr;
	PTX::Register<PTX::UInt32Type> *m_rightIndex = nullptr;
};

template<PTX::Bits B>
class OrderGenerator : public Generator
{
public:
	OrderGenerator(Builder& builder, OrderMode mode) : Generator(builder), m_mode(mode) {}

	std::string Name() const override { return "OrderGenerator"; }

	void Generate(const std::vector<const HorseIR::Operand *>& arguments)
	{
		const auto indexSize = HorseIR::TypeUtils::GetBitSize(arguments.at(0)->GetType());
		const auto dataSize = HorseIR::TypeUtils::GetBitSize(arguments.at(1)->GetType());
		const auto size = (indexSize + dataSize) / 8;

		auto& targetOptions = m_builder.GetTargetOptions();
		const auto sharedSize = targetOptions.SharedMemorySize;

		if (size * 2048 <= sharedSize)
		{
			Generate<1024>(arguments);
		}
		else if (size * 1024 <= sharedSize)
		{
			Generate<512>(arguments);
		}
		else if (size * 512 <= sharedSize)
		{
			Generate<256>(arguments);
		}
		else if (size * 256 <= sharedSize)
		{
			Generate<128>(arguments);
		}
		else
		{
			Error("shared memory size less than 128 threads");
		}
	}

	template<unsigned int SORT_CACHE_SIZE>
	void Generate(const std::vector<const HorseIR::Operand *>& arguments)
	{
		auto resources = this->m_builder.GetLocalResources();

		const auto indexArgument = arguments.at(0);
		const auto dataArgument = arguments.at(1);
		const auto orderLiteral = HorseIR::LiteralUtils<std::int8_t>::GetLiteral(arguments.at(2));

		DataIndexGenerator<B> indexGenerator(this->m_builder);
		auto index = indexGenerator.GenerateDataIndex();

		// Add stage/substage parameters and load values

		ParameterGenerator<B> parameterGenerator(this->m_builder);
		auto sortStageParameter = parameterGenerator.template GenerateConstant<PTX::UInt32Type>(NameUtils::SortStage);
		auto sortSubstageParameter = parameterGenerator.template GenerateConstant<PTX::UInt32Type>(NameUtils::SortSubstage);

		ValueLoadGenerator<B, PTX::UInt32Type> valueLoadGenerator(this->m_builder);
		auto stage = valueLoadGenerator.GenerateConstant(sortStageParameter);
		auto substage = valueLoadGenerator.GenerateConstant(sortSubstageParameter);

		auto stageFunction = [&]()
		{
			// Compute the size of each bitonic sequence in this stage
			//   sequenceSize = 2^(stage + 1)

			auto temp_stageP1 = resources->template AllocateTemporary<PTX::UInt32Type>();
			auto temp2 = resources->template AllocateTemporary<PTX::UInt32Type>();
			auto sequenceSize = resources->template AllocateTemporary<PTX::UInt32Type>();

			this->m_builder.AddStatement(new PTX::AddInstruction<PTX::UInt32Type>(temp_stageP1, stage, new PTX::UInt32Value(1)));
			this->m_builder.AddStatement(new PTX::MoveInstruction<PTX::UInt32Type>(temp2, new PTX::UInt32Value(1)));
			this->m_builder.AddStatement(new PTX::ShiftLeftInstruction<PTX::Bit32Type>(
				new PTX::Bit32RegisterAdapter<PTX::UIntType>(sequenceSize),
				new PTX::Bit32Adapter<PTX::UIntType>(temp2),
				temp_stageP1
			));

			// Compute the sequence index of this thread. We allocate threads for half the sequence size as each thread will perform 1 swap
			//   sequenceIndex = (index / (sequenceSize >> 1))

			auto temp_halfSequence = resources->template AllocateTemporary<PTX::UInt32Type>();
			auto sequenceIndex = resources->template AllocateTemporary<PTX::UInt32Type>();

			this->m_builder.AddStatement(new PTX::ShiftRightInstruction<PTX::Bit32Type>(
				new PTX::Bit32RegisterAdapter<PTX::UIntType>(temp_halfSequence),
				new PTX::Bit32Adapter<PTX::UIntType>(sequenceSize),
				new PTX::UInt32Value(1)
			));

			this->m_builder.AddStatement(new PTX::ShiftRightInstruction<PTX::Bit32Type>(
				new PTX::Bit32RegisterAdapter<PTX::UIntType>(sequenceIndex),
				new PTX::Bit32Adapter<PTX::UIntType>(index),
				stage
			));

			auto temp_halfSequenceM1 = resources->template AllocateTemporary<PTX::UInt32Type>();
			auto temp4 = resources->template AllocateTemporary<PTX::UInt32Type>();

			this->m_builder.AddStatement(new PTX::SubtractInstruction<PTX::UInt32Type>(temp_halfSequenceM1, temp_halfSequence, new PTX::UInt32Value(1)));
			this->m_builder.AddStatement(new PTX::AndInstruction<PTX::Bit32Type>(
				new PTX::Bit32RegisterAdapter<PTX::UIntType>(temp4),
				new PTX::Bit32Adapter<PTX::UIntType>(index),
				new PTX::Bit32Adapter<PTX::UIntType>(temp_halfSequenceM1)
			));

			// Compute the sequence start index for this thread
			//   sequenceStart = sequenceIndex * sequenceSize = sequenceIndex * 2^(stage + 1)

			auto sequenceStart = resources->template AllocateTemporary<PTX::UInt32Type>();

			// Multiplication by power-of-2 number

			this->m_builder.AddStatement(new PTX::ShiftLeftInstruction<PTX::Bit32Type>(
				new PTX::Bit32RegisterAdapter<PTX::UIntType>(sequenceStart),
				new PTX::Bit32Adapter<PTX::UIntType>(sequenceIndex),
				temp_stageP1
			));

			// Bitonic sequence direction

			auto temp5 = resources->template AllocateTemporary<PTX::Bit32Type>();
			auto subsequenceDirection = resources->template AllocateTemporary<PTX::PredicateType>();

			this->m_builder.AddStatement(new PTX::AndInstruction<PTX::Bit32Type>(
				temp5, new PTX::Bit32Adapter<PTX::UIntType>(sequenceIndex), new PTX::Bit32Adapter<PTX::UIntType>(new PTX::UInt32Value(1))
			));
			this->m_builder.AddStatement(new PTX::SetPredicateInstruction<PTX::Bit32Type>(
				subsequenceDirection, temp5, new PTX::Bit32Adapter<PTX::UIntType>(new PTX::UInt32Value(1)), PTX::Bit32Type::ComparisonOperator::NotEqual
			));

			auto substageFunction = [&]()
			{
				// Compute the size of each substage
				//   subsequenceSize = sequenceSize >> substage

				auto subsequenceSize = resources->template AllocateTemporary<PTX::UInt32Type>();

				this->m_builder.AddStatement(new PTX::ShiftRightInstruction<PTX::Bit32Type>(
					new PTX::Bit32RegisterAdapter<PTX::UIntType>(subsequenceSize),
					new PTX::Bit32Adapter<PTX::UIntType>(sequenceSize),
					substage
				));

				// Compute the index of the substage, again half the number of threads are active
				//   subsequenceIndex = (index % (sequenceSize >> 1)) / (subsequenceSize >> 1);

				auto temp_halfSubsequence = resources->template AllocateTemporary<PTX::UInt32Type>();
				auto subsequenceIndex = resources->template AllocateTemporary<PTX::UInt32Type>();

				this->m_builder.AddStatement(new PTX::ShiftRightInstruction<PTX::Bit32Type>(
					new PTX::Bit32RegisterAdapter<PTX::UIntType>(temp_halfSubsequence),
					new PTX::Bit32Adapter<PTX::UIntType>(subsequenceSize),
					new PTX::UInt32Value(1)
				));

				// Fancy division by power-of-2

				auto temp_gap = resources->template AllocateTemporary<PTX::UInt32Type>();

				this->m_builder.AddStatement(new PTX::SubtractInstruction<PTX::UInt32Type>(temp_gap, stage, substage));
				this->m_builder.AddStatement(new PTX::ShiftRightInstruction<PTX::Bit32Type>(
					new PTX::Bit32RegisterAdapter<PTX::UIntType>(subsequenceIndex),
					new PTX::Bit32Adapter<PTX::UIntType>(temp4),
					temp_gap
				));
                                
				// Compute the subsequence start index for this thread
				//   subsequenceStart = sequenceStart + (subsequenceIndex * subsequenceSize)

				auto subsequenceStart = resources->template AllocateTemporary<PTX::UInt32Type>();

				//TODO: Optimize
				this->m_builder.AddStatement(new PTX::MADInstruction<PTX::UInt32Type>(
					subsequenceStart, subsequenceIndex, subsequenceSize, sequenceStart, PTX::HalfModifier<PTX::UInt32Type>::Half::Lower
				));

				// Index of the thread in *its* substage
				//   subsequenceLocalIndex = index % (subsequenceSize >> 2)

				auto subsequenceLocalIndex = resources->template AllocateTemporary<PTX::UInt32Type>();
				auto temp_halfSubsequenceM1 = resources->template AllocateTemporary<PTX::UInt32Type>();

				this->m_builder.AddStatement(new PTX::SubtractInstruction<PTX::UInt32Type>(temp_halfSubsequenceM1, temp_halfSubsequence, new PTX::UInt32Value(1)));
				this->m_builder.AddStatement(new PTX::AndInstruction<PTX::Bit32Type>(
					new PTX::Bit32RegisterAdapter<PTX::UIntType>(subsequenceLocalIndex),
					new PTX::Bit32Adapter<PTX::UIntType>(index),
					new PTX::Bit32Adapter<PTX::UIntType>(temp_halfSubsequenceM1)
				));

				// Compute the indices of the left and right data items
				//   leftIndex = subsequenceStart + subsequenceLocalIndex
				//   rightIndex = leftIndex + (subsequenceSize >> 1)
				//
				// If shared, mod SORT_CACHE_SIZE * 2

				auto leftIndex = resources->template AllocateTemporary<PTX::UInt32Type>();
				auto rightIndex = resources->template AllocateTemporary<PTX::UInt32Type>();

				this->m_builder.AddStatement(new PTX::AddInstruction<PTX::UInt32Type>(leftIndex, subsequenceStart, subsequenceLocalIndex));
				this->m_builder.AddStatement(new PTX::AddInstruction<PTX::UInt32Type>(rightIndex, leftIndex, temp_halfSubsequence));

				if (m_mode == OrderMode::Shared)
				{
					this->m_builder.AddStatement(new PTX::AndInstruction<PTX::Bit32Type>(
						new PTX::Bit32RegisterAdapter<PTX::UIntType>(leftIndex),
						new PTX::Bit32Adapter<PTX::UIntType>(leftIndex),
						new PTX::Bit32Adapter<PTX::UIntType>(new PTX::UInt32Value(SORT_CACHE_SIZE * 2 - 1))
					));
					this->m_builder.AddStatement(new PTX::AndInstruction<PTX::Bit32Type>(
						new PTX::Bit32RegisterAdapter<PTX::UIntType>(rightIndex),
						new PTX::Bit32Adapter<PTX::UIntType>(rightIndex),
						new PTX::Bit32Adapter<PTX::UIntType>(new PTX::UInt32Value(SORT_CACHE_SIZE * 2 - 1))
					));
				}

				// Load the left and right values

				OrderLoadGenerator<B, SORT_CACHE_SIZE> loadGenerator(this->m_builder, m_mode);
				loadGenerator.Generate(dataArgument, leftIndex, rightIndex);

				// Generate the if-else structure for the sort order

				auto swapPredicate = resources->template AllocateTemporary<PTX::PredicateType>();
				this->m_builder.AddStatement(new PTX::MoveInstruction<PTX::PredicateType>(swapPredicate, new PTX::BoolValue(false)));

				this->m_builder.AddIfElseStatement("ORDER", [&]()
				{
					return std::make_tuple(subsequenceDirection, false);
				},
				[&]()
				{
					// True branch (ascending sequence)

					OrderComparisonGenerator<B> ascendingGenerator(this->m_builder, OrderComparisonGenerator<B>::Order::Ascending, swapPredicate);
					ascendingGenerator.Generate(dataArgument, orderLiteral);
				},
				[&]()
				{
					// Else branch (descending sequence)

					OrderComparisonGenerator<B> descendingGenerator(this->m_builder, OrderComparisonGenerator<B>::Order::Descending, swapPredicate);
					descendingGenerator.Generate(dataArgument, orderLiteral);
				});

				this->m_builder.AddIfStatement("SWAP", [&]()
				{
					return std::make_tuple(swapPredicate, true);
				},
				[&]()
				{
					// Swap if needed! Only load index if swapping

					loadGenerator.Generate(indexArgument, leftIndex, rightIndex);

					OrderSwapGenerator<B, SORT_CACHE_SIZE> swapGenerator(this->m_builder, m_mode);
					swapGenerator.Generate(indexArgument, dataArgument, leftIndex, rightIndex);
				});
			};

			if (m_mode == OrderMode::Shared)
			{
				// Setup inner loop for iterating substages

				this->m_builder.AddDoWhileLoop("SUBSTAGE", [&](Builder::LoopContext& loopContext)
				{
					substageFunction();

					// Synchronize shared memory

					BarrierGenerator<B> barrierGenerator(this->m_builder);
					barrierGenerator.Generate();

					// End of inner loop

					this->m_builder.AddStatement(new PTX::AddInstruction<PTX::UInt32Type>(substage, substage, new PTX::UInt32Value(1)));

					auto substagePredicate = resources->template AllocateTemporary<PTX::PredicateType>();
					this->m_builder.AddStatement(new PTX::SetPredicateInstruction<PTX::UInt32Type>(
						substagePredicate, substage, stage, PTX::UInt32Type::ComparisonOperator::LessEqual
					));
					return std::make_tuple(substagePredicate, false);
				});
			}
			else
			{
				substageFunction();
			}
		};

		if (m_mode == OrderMode::Shared)
		{
			// Add the special parameter for the number of stages

			auto sortNumStagesParameter = parameterGenerator.template GenerateConstant<PTX::UInt32Type>(NameUtils::SortNumStages);
			auto numStages = valueLoadGenerator.GenerateConstant(sortNumStagesParameter);

			ThreadIndexGenerator<B> indexGenerator(this->m_builder);
			auto blockIndex = indexGenerator.GenerateBlockIndex();
			auto localIndex = indexGenerator.GenerateLocalIndex();

			//TODO: Optimize
			auto sharedIndex = resources->template AllocateTemporary<PTX::UInt32Type>();
			this->m_builder.AddStatement(new PTX::MADInstruction<PTX::UInt32Type>(
				sharedIndex, new PTX::UInt32Value(SORT_CACHE_SIZE * 2), blockIndex, localIndex, PTX::HalfModifier<PTX::UInt32Type>::Half::Lower
			));

			InternalCacheGenerator_Load<B, SORT_CACHE_SIZE, 2> cacheGenerator(this->m_builder);
			cacheGenerator.SetBoundsCheck(false);

			cacheGenerator.SetSynchronize(false);
			cacheGenerator.Generate(indexArgument, sharedIndex);

			cacheGenerator.SetSynchronize(true);
			cacheGenerator.Generate(dataArgument, sharedIndex);

			// Initialize stage and bound

			auto totalStages = resources->template AllocateTemporary<PTX::UInt32Type>();
			this->m_builder.AddStatement(new PTX::AddInstruction<PTX::UInt32Type>(totalStages, stage, numStages));

			// Setup outer loop for iterating stages

			this->m_builder.AddDoWhileLoop("STAGE", [&](Builder::LoopContext& loopContext)
			{
				stageFunction();

				// End of outer loop

				this->m_builder.AddStatement(new PTX::MoveInstruction<PTX::UInt32Type>(substage, new PTX::UInt32Value(0)));
				this->m_builder.AddStatement(new PTX::AddInstruction<PTX::UInt32Type>(stage, stage, new PTX::UInt32Value(1)));

				auto stagePredicate = resources->template AllocateTemporary<PTX::PredicateType>();
				this->m_builder.AddStatement(new PTX::SetPredicateInstruction<PTX::UInt32Type>(
					stagePredicate, stage, totalStages, PTX::UInt32Type::ComparisonOperator::Less
				));
				return std::make_tuple(stagePredicate, false);
			});

			// Store cached data back in its entirety

			InternalCacheGenerator_Store<B, SORT_CACHE_SIZE, 2> cacheStoreGenerator(this->m_builder);
			cacheStoreGenerator.Generate(indexArgument, sharedIndex);
			cacheStoreGenerator.Generate(dataArgument, sharedIndex);
		}
		else
		{
			stageFunction();
		}
	}

private:
	OrderMode m_mode = OrderMode::Shared;
};

}
}
