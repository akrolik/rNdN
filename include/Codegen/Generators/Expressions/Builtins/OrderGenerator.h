#pragma once

#include "Codegen/Generators/Generator.h"
#include "HorseIR/Traversal/ConstVisitor.h"

#include "Codegen/Builder.h"
#include "Codegen/Generators/Data/ParameterGenerator.h"
#include "Codegen/Generators/Data/ValueLoadGenerator.h"
#include "Codegen/Generators/Expressions/OperandGenerator.h"
#include "Codegen/Generators/Expressions/Builtins/InternalCacheGenerator.h"
#include "Codegen/Generators/Indexing/AddressGenerator.h"
#include "Codegen/Generators/Indexing/DataIndexGenerator.h"
#include "Codegen/Generators/Indexing/ThreadIndexGenerator.h"

#include "HorseIR/Tree/Tree.h"
#include "HorseIR/Utils/LiteralUtils.h"

#include "PTX/PTX.h"

namespace Codegen {

constexpr auto SORT_CACHE_SIZE = 1024u;

enum class OrderMode {
	Shared,
	Global
};

template<PTX::Bits B>
class OrderLoadGenerator : public Generator, public HorseIR::ConstVisitor
{
public:
	OrderLoadGenerator(Builder& builder, OrderMode mode) : Generator(builder), m_mode(mode) {}

	std::string Name() const override { return "OrderLoadGenerator"; }

	void Generate(const HorseIR::Operand *dataArgument, const PTX::Register<PTX::UInt32Type> *leftIndex, const PTX::Register<PTX::UInt32Type> *rightIndex)
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

		if (const auto listShape = Analysis::ShapeUtils::GetShape<Analysis::ListShape>(shape))
		{
			if (const auto size = Analysis::ShapeUtils::GetSize<Analysis::Shape::ConstantSize>(listShape->GetListSize()))
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

	const PTX::Register<PTX::UInt32Type> *m_leftIndex = nullptr;
	const PTX::Register<PTX::UInt32Type> *m_rightIndex = nullptr;
};

template<PTX::Bits B>
class OrderComparisonGenerator : public Generator, public HorseIR::ConstVisitor
{
public:
	enum class Order {
		Ascending,
		Descending
	};

	OrderComparisonGenerator(Builder& builder, Order sequenceOrder, const PTX::Label *swapLabel, const PTX::Label *endLabel)
		: Generator(builder), m_sequenceOrder(sequenceOrder), m_swapLabel(swapLabel), m_endLabel(endLabel) {}

	std::string Name() const override { return "OrderComparisonGenerator"; }

	void Generate(const HorseIR::Operand *dataArgument, const HorseIR::TypedVectorLiteral<std::int8_t> *orderLiteral)
	{
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

		if (const auto listShape = Analysis::ShapeUtils::GetShape<Analysis::ListShape>(shape))
		{
			if (const auto size = Analysis::ShapeUtils::GetSize<Analysis::Shape::ConstantSize>(listShape->GetListSize()))
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
		GenerateComparison<T>(identifier, true, index);
	}

private:
	template<class T>
	void GenerateComparison(const HorseIR::Identifier *identifier, bool isCell = false, unsigned int index = 0)
	{
		if constexpr(std::is_same<T, PTX::PredicateType>::value || std::is_same<T, PTX::Int8Type>::value)
		{
			GenerateComparison<PTX::Int16Type>(identifier, isCell, index);
		}
		else
		{
			OperandGenerator<B, T> operandGenerator(this->m_builder);
			auto leftValue = operandGenerator.GenerateOperand(identifier, nullptr, "left", index);
			auto rightValue = operandGenerator.GenerateOperand(identifier, nullptr, "right", index);

			auto resources = this->m_builder.GetLocalResources();
			auto predicateSwap = resources->template AllocateTemporary<PTX::PredicateType>();

			auto dataOrder = (m_orderLiteral->GetValue((m_orderLiteral->GetCount() == 1 ? 0 : index))) ? Order::Ascending : Order::Descending;
			if (m_sequenceOrder == dataOrder)
			{
				this->m_builder.AddStatement(new PTX::SetPredicateInstruction<T>(predicateSwap, leftValue, rightValue, T::ComparisonOperator::Less));
			}
			else
			{
				this->m_builder.AddStatement(new PTX::SetPredicateInstruction<T>(predicateSwap, leftValue, rightValue, T::ComparisonOperator::Greater));
			}

			// Branch if the predicate is true

			this->m_builder.AddStatement(new PTX::BranchInstruction(m_swapLabel, predicateSwap));
			
			// Check for the next branch

			auto predicateEqual = resources->template AllocateTemporary<PTX::PredicateType>();

			this->m_builder.AddStatement(new PTX::SetPredicateInstruction<T>(predicateEqual, leftValue, rightValue, T::ComparisonOperator::NotEqual));
			this->m_builder.AddStatement(new PTX::BranchInstruction(m_endLabel, predicateEqual));
		}
	}

private:
	const PTX::Label *m_swapLabel = nullptr;
	const PTX::Label *m_endLabel = nullptr;

	const HorseIR::TypedVectorLiteral<std::int8_t> *m_orderLiteral = nullptr;
	Order m_sequenceOrder;
};

template<PTX::Bits B>
class OrderSwapGenerator : public Generator, public HorseIR::ConstVisitor
{
public:
	OrderSwapGenerator(Builder& builder, OrderMode mode) : Generator(builder), m_mode(mode) {}

	std::string Name() const override { return "OrderSwapGenerator"; }

	void Generate(const HorseIR::Operand *index, const HorseIR::Operand *dataArgument, const PTX::Register<PTX::UInt32Type> *leftIndex, const PTX::Register<PTX::UInt32Type> *rightIndex)
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

		if (const auto listShape = Analysis::ShapeUtils::GetShape<Analysis::ListShape>(shape))
		{
			if (const auto size = Analysis::ShapeUtils::GetSize<Analysis::Shape::ConstantSize>(listShape->GetListSize()))
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

	const PTX::Register<PTX::UInt32Type> *m_leftIndex = nullptr;
	const PTX::Register<PTX::UInt32Type> *m_rightIndex = nullptr;
};

template<PTX::Bits B>
class OrderGenerator : public Generator
{
public:
	OrderGenerator(Builder& builder, OrderMode mode) : Generator(builder), m_mode(mode) {}

	std::string Name() const override { return "OrderGenerator"; }

	void Generate(const std::vector<HorseIR::Operand *>& arguments)
	{
		auto resources = this->m_builder.GetLocalResources();

		const auto indexArgument = arguments.at(0);
		const auto dataArgument = arguments.at(1);
		const auto orderLiteral = HorseIR::LiteralUtils<std::int8_t>::GetLiteral(arguments.at(2));

		const PTX::Register<PTX::UInt32Type> *sharedIndex = nullptr;

		const PTX::Register<PTX::UInt32Type> *stage = nullptr;
		const PTX::Register<PTX::UInt32Type> *substage = nullptr;
		const PTX::Register<PTX::UInt32Type> *startSubstage = nullptr;
		const PTX::Register<PTX::UInt32Type> *totalStages = nullptr;

		const PTX::Label *stageStartLabel = nullptr;
		const PTX::Label *stageEndLabel = nullptr;

		const PTX::Label *substageStartLabel = nullptr;
		const PTX::Label *substageEndLabel = nullptr;

		DataIndexGenerator<B> indexGenerator(this->m_builder);
		auto index = indexGenerator.GenerateDataIndex();

		switch (m_mode)
		{
			case OrderMode::Shared:
			{
				// Add the special parameters for sorting (start_stage, start_substage, num_stages) and load the values

				ParameterGenerator<B> parameterGenerator(this->m_builder);
				auto sortStartStageParameter = parameterGenerator.template GenerateConstant<PTX::UInt32Type>(NameUtils::SortStartStage);
				auto sortStartSubstageParameter = parameterGenerator.template GenerateConstant<PTX::UInt32Type>(NameUtils::SortStartSubstage);
				auto sortNumStagesParameter = parameterGenerator.template GenerateConstant<PTX::UInt32Type>(NameUtils::SortNumStages);

				ValueLoadGenerator<B, PTX::UInt32Type> valueLoadGenerator(this->m_builder);
				auto startStage = valueLoadGenerator.GenerateConstant(sortStartStageParameter);
				startSubstage = valueLoadGenerator.GenerateConstant(sortStartSubstageParameter);
				auto numStages = valueLoadGenerator.GenerateConstant(sortNumStagesParameter);

				ThreadIndexGenerator<B> indexGenerator(this->m_builder);
				auto blockIndex = indexGenerator.GenerateBlockIndex();
				auto localIndex = indexGenerator.GenerateLocalIndex();

				sharedIndex = resources->template AllocateTemporary<PTX::UInt32Type>();
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

				totalStages = resources->template AllocateTemporary<PTX::UInt32Type>();
				this->m_builder.AddStatement(new PTX::AddInstruction<PTX::UInt32Type>(totalStages, startStage, numStages));

				stage = resources->template AllocateTemporary<PTX::UInt32Type>();
				this->m_builder.AddStatement(new PTX::MoveInstruction<PTX::UInt32Type>(stage, startStage));

				// Setup outer loop for iterating stages

				stageStartLabel = this->m_builder.CreateLabel("STAGE_START");
				stageEndLabel = this->m_builder.CreateLabel("STAGE_END");
				auto stagePredicate = resources->template AllocateTemporary<PTX::PredicateType>();

				this->m_builder.AddStatement(new PTX::SetPredicateInstruction<PTX::UInt32Type>(
					stagePredicate, stage, totalStages, PTX::UInt32Type::ComparisonOperator::GreaterEqual
				));
				this->m_builder.AddStatement(new PTX::BranchInstruction(stageEndLabel, stagePredicate, false, true));
				this->m_builder.AddStatement(stageStartLabel);

				break;
			}
			case OrderMode::Global:
			{
				// Add the special parameters for sorting (stage and substage) and load the values

				ParameterGenerator<B> parameterGenerator(this->m_builder);
				auto sortStageParameter = parameterGenerator.template GenerateConstant<PTX::UInt32Type>(NameUtils::SortStage);
				auto sortSubstageParameter = parameterGenerator.template GenerateConstant<PTX::UInt32Type>(NameUtils::SortSubstage);

				ValueLoadGenerator<B, PTX::UInt32Type> valueLoadGenerator(this->m_builder);
				stage = valueLoadGenerator.GenerateConstant(sortStageParameter);
				substage = valueLoadGenerator.GenerateConstant(sortSubstageParameter);

				break;
			}
		}
		// Compute the size of each bitonic sequence in this stage
		//   sequenceSize = 2^(stage + 1)

		auto temp1 = resources->template AllocateTemporary<PTX::UInt32Type>();
		auto temp2 = resources->template AllocateTemporary<PTX::UInt32Type>();
		auto sequenceSize = resources->template AllocateTemporary<PTX::UInt32Type>();

		this->m_builder.AddStatement(new PTX::AddInstruction<PTX::UInt32Type>(temp1, stage, new PTX::UInt32Value(1)));
		this->m_builder.AddStatement(new PTX::MoveInstruction<PTX::UInt32Type>(temp2, new PTX::UInt32Value(1)));
		this->m_builder.AddStatement(new PTX::ShiftLeftInstruction<PTX::Bit32Type>(
			new PTX::Bit32RegisterAdapter<PTX::UIntType>(sequenceSize),
			new PTX::Bit32Adapter<PTX::UIntType>(temp2),
			temp1
		));

		// Compute the sequence index of this thread. We allocate threads for half the sequence size as each thread will perform 1 swap
		//   sequenceIndex = (index / (sequenceSize / 2))

		auto temp_halfSequence = resources->template AllocateTemporary<PTX::UInt32Type>();
		auto sequenceIndex = resources->template AllocateTemporary<PTX::UInt32Type>();

		this->m_builder.AddStatement(new PTX::ShiftRightInstruction<PTX::Bit32Type>(
			new PTX::Bit32RegisterAdapter<PTX::UIntType>(temp_halfSequence),
			new PTX::Bit32Adapter<PTX::UIntType>(sequenceSize),
			new PTX::UInt32Value(1)
		));
		this->m_builder.AddStatement(new PTX::DivideInstruction<PTX::UInt32Type>(sequenceIndex, index, temp_halfSequence));

		// Compute the sequence start index for this thread
		//   sequenceStart = sequenceindex * sequenceSize

		auto sequenceStart = resources->template AllocateTemporary<PTX::UInt32Type>();

		this->m_builder.AddStatement(new PTX::MultiplyInstruction<PTX::UInt32Type>(
			sequenceStart, sequenceIndex, sequenceSize, PTX::HalfModifier<PTX::UInt32Type>::Half::Lower
		));

		if (m_mode == OrderMode::Shared)
		{
			// Initialize substage

			substage = resources->template AllocateTemporary<PTX::UInt32Type>();
			this->m_builder.AddStatement(new PTX::MoveInstruction<PTX::UInt32Type>(substage, startSubstage));

			// Setup inner loop for iterating substages

			substageStartLabel = this->m_builder.CreateLabel("SUBSTAGE_START");
			substageEndLabel = this->m_builder.CreateLabel("SUBSTAGE_END");
			auto substagePredicate = resources->template AllocateTemporary<PTX::PredicateType>();

			this->m_builder.AddStatement(new PTX::SetPredicateInstruction<PTX::UInt32Type>(
				substagePredicate, substage, stage, PTX::UInt32Type::ComparisonOperator::Greater
			));
			this->m_builder.AddStatement(new PTX::BranchInstruction(substageEndLabel, substagePredicate, false, true));
			this->m_builder.AddStatement(substageStartLabel);
		}

		// Compute the size of each substage
		//   subsequenceSize = sequenceSize >> substage

		auto subsequenceSize = resources->template AllocateTemporary<PTX::UInt32Type>();

		this->m_builder.AddStatement(new PTX::ShiftRightInstruction<PTX::Bit32Type>(
			new PTX::Bit32RegisterAdapter<PTX::UIntType>(subsequenceSize),
			new PTX::Bit32Adapter<PTX::UIntType>(sequenceSize),
			substage
		));

		// Compute the index of the substage, again half the number of threads are active
		//   subsequenceIndex = (index % (sequenceSize / 2)) / (subsequenceSize / 2);

		auto temp4 = resources->template AllocateTemporary<PTX::UInt32Type>();
		auto temp_halfSubsequence = resources->template AllocateTemporary<PTX::UInt32Type>();
		auto subsequenceIndex = resources->template AllocateTemporary<PTX::UInt32Type>();

		this->m_builder.AddStatement(new PTX::RemainderInstruction<PTX::UInt32Type>(temp4, index, temp_halfSequence));
		this->m_builder.AddStatement(new PTX::ShiftRightInstruction<PTX::Bit32Type>(
			new PTX::Bit32RegisterAdapter<PTX::UIntType>(temp_halfSubsequence),
			new PTX::Bit32Adapter<PTX::UIntType>(subsequenceSize),
			new PTX::UInt32Value(1)
		));
		this->m_builder.AddStatement(new PTX::DivideInstruction<PTX::UInt32Type>(subsequenceIndex, temp4, temp_halfSubsequence));

		// Compute the subsequence start index for this thread
		//   subsequenceStart = sequenceStart + (subsequenceIndex * subsequenceSize)

		auto subsequenceStart = resources->template AllocateTemporary<PTX::UInt32Type>();

		this->m_builder.AddStatement(new PTX::MADInstruction<PTX::UInt32Type>(
			subsequenceStart, subsequenceIndex, subsequenceSize, sequenceStart, PTX::HalfModifier<PTX::UInt32Type>::Half::Lower
		));

		// Index of the thread in *its* substage
		//   subsequenceLocalIndex = index % (subsequenceSize / 2)

		auto subsequenceLocalIndex = resources->template AllocateTemporary<PTX::UInt32Type>();

		this->m_builder.AddStatement(new PTX::RemainderInstruction<PTX::UInt32Type>(subsequenceLocalIndex, index, temp_halfSubsequence));

		// Compute the indices of the left and right data items
		//   leftIndex = subsequenceStart + subsequenceLocalIndex
		//   rightIndex = leftIndex + (subsequenceSize / 2)
		//
		// If shared, mod SORT_CACHE_SIZE * 2

		auto leftIndex = resources->template AllocateTemporary<PTX::UInt32Type>();
		auto rightIndex = resources->template AllocateTemporary<PTX::UInt32Type>();

		this->m_builder.AddStatement(new PTX::AddInstruction<PTX::UInt32Type>(leftIndex, subsequenceStart, subsequenceLocalIndex));
		this->m_builder.AddStatement(new PTX::AddInstruction<PTX::UInt32Type>(rightIndex, leftIndex, temp_halfSubsequence));

		if (m_mode == OrderMode::Shared)
		{
			this->m_builder.AddStatement(new PTX::RemainderInstruction<PTX::UInt32Type>(leftIndex, leftIndex, new PTX::UInt32Value(SORT_CACHE_SIZE * 2)));
			this->m_builder.AddStatement(new PTX::RemainderInstruction<PTX::UInt32Type>(rightIndex, rightIndex, new PTX::UInt32Value(SORT_CACHE_SIZE * 2)));
		}

		// Load the left and right values

		OrderLoadGenerator<B> loadGenerator(this->m_builder, m_mode);
		loadGenerator.Generate(dataArgument, leftIndex, rightIndex);

		// Generate the if-else structure for the sort order

		auto temp5 = resources->template AllocateTemporary<PTX::Bit32Type>();
		auto subsequenceDirection = resources->template AllocateTemporary<PTX::PredicateType>();

		this->m_builder.AddStatement(new PTX::AndInstruction<PTX::Bit32Type>(
			temp5, new PTX::Bit32Adapter<PTX::UIntType>(sequenceIndex), new PTX::Bit32Adapter<PTX::UIntType>(new PTX::UInt32Value(1))
		));
		this->m_builder.AddStatement(new PTX::SetPredicateInstruction<PTX::Bit32Type>(
			subsequenceDirection, temp5, new PTX::Bit32Adapter<PTX::UIntType>(new PTX::UInt32Value(1)), PTX::Bit32Type::ComparisonOperator::NotEqual
		));

		auto elseLabel = this->m_builder.CreateLabel("ELSE");
		auto swapLabel = this->m_builder.CreateLabel("SWAP");
		auto endLabel = this->m_builder.CreateLabel("END");

		this->m_builder.AddStatement(new PTX::BranchInstruction(elseLabel, subsequenceDirection));
		this->m_builder.AddStatement(new PTX::BlankStatement());

		// True branch (ascending sequence)

		OrderComparisonGenerator<B> ascendingGenerator(this->m_builder, OrderComparisonGenerator<B>::Order::Ascending, swapLabel, endLabel);
		ascendingGenerator.Generate(dataArgument, orderLiteral);

		this->m_builder.AddStatement(new PTX::BranchInstruction(endLabel, true));

		// Else branch (descending sequence)

		this->m_builder.AddStatement(elseLabel);
		this->m_builder.AddStatement(new PTX::BlankStatement());

		OrderComparisonGenerator<B> descendingGenerator(this->m_builder, OrderComparisonGenerator<B>::Order::Descending, swapLabel, endLabel);
		descendingGenerator.Generate(dataArgument, orderLiteral);

		this->m_builder.AddStatement(new PTX::BranchInstruction(endLabel, true));

		// Swap if needed!

		this->m_builder.AddStatement(swapLabel);

		// Only load index if needed

		loadGenerator.Generate(indexArgument, leftIndex, rightIndex);

		OrderSwapGenerator<B> swapGenerator(this->m_builder, m_mode);
		swapGenerator.Generate(indexArgument, dataArgument, leftIndex, rightIndex);

		// Finally, we end the order

		this->m_builder.AddStatement(endLabel);

		if (m_mode == OrderMode::Shared)
		{
			// Synchronize shared memory

			BarrierGenerator<B> barrierGenerator(this->m_builder);
			barrierGenerator.Generate();

			// End of inner loop

			this->m_builder.AddStatement(new PTX::AddInstruction<PTX::UInt32Type>(substage, substage, new PTX::UInt32Value(1)));

			auto substagePredicate = resources->template AllocateTemporary<PTX::PredicateType>();
			this->m_builder.AddStatement(new PTX::SetPredicateInstruction<PTX::UInt32Type>(
				substagePredicate, substage, stage, PTX::UInt32Type::ComparisonOperator::LessEqual
			));

			this->m_builder.AddStatement(new PTX::BranchInstruction(substageStartLabel, substagePredicate, false, true));
			this->m_builder.AddStatement(substageEndLabel);

			// End of outer loop

			this->m_builder.AddStatement(new PTX::AddInstruction<PTX::UInt32Type>(stage, stage, new PTX::UInt32Value(1)));

			auto stagePredicate = resources->template AllocateTemporary<PTX::PredicateType>();
			this->m_builder.AddStatement(new PTX::SetPredicateInstruction<PTX::UInt32Type>(
				stagePredicate, stage, totalStages, PTX::UInt32Type::ComparisonOperator::Less
			));

			this->m_builder.AddStatement(new PTX::BranchInstruction(stageStartLabel, stagePredicate, false, true));
			this->m_builder.AddStatement(stageEndLabel);

			// Store cached data back in its entirety

			InternalCacheGenerator_Store<B, SORT_CACHE_SIZE, 2> cacheStoreGenerator(this->m_builder);
			cacheStoreGenerator.Generate(indexArgument, sharedIndex);
			cacheStoreGenerator.Generate(dataArgument, sharedIndex);
		}
	}

private:
	OrderMode m_mode = OrderMode::Shared;
};

}
