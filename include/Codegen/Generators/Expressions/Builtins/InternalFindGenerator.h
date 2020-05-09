#pragma once

#include <utility>

#include "HorseIR/Traversal/ConstVisitor.h"
#include "Codegen/Generators/Expressions/Builtins/BuiltinGenerator.h"

#include "Codegen/Builder.h"
#include "Codegen/Generators/Expressions/ConversionGenerator.h"
#include "Codegen/Generators/Expressions/OperandCompressionGenerator.h"
#include "Codegen/Generators/Expressions/OperandGenerator.h"
#include "Codegen/Generators/Expressions/Builtins/ComparisonGenerator.h"
#include "Codegen/Generators/Indexing/AddressGenerator.h"
#include "Codegen/Generators/Indexing/DataSizeGenerator.h"
#include "Codegen/Generators/Indexing/ThreadIndexGenerator.h"
#include "Codegen/Generators/Synchronization/BarrierGenerator.h"
#include "Codegen/NameUtils.h"

#include "HorseIR/Tree/Tree.h"

#include "PTX/PTX.h"

namespace Codegen {

enum class FindOperation {
	Member,
	Index,
	Count,
	Indexes
};

constexpr auto FIND_BLOCK_SIZE = 1024u;

template<PTX::Bits B>
class InternalFindGenerator_Init : public Generator
{
public:
	using Generator::Generator; 

	std::string Name() const override { return "InternalFindGenerator_Init"; }

	void Generate(const HorseIR::Operand *dataX, const HorseIR::Identifier *dataY)
	{
		DispatchType(*this, dataX->GetType(), dataX, dataY);
	}

	template<class T>
	void GenerateVector(const HorseIR::Operand *dataX, const HorseIR::Identifier *dataY)
	{
		GenerateInit<T>(dataX, dataY);
	}

	template<class T>
	void GenerateList(const HorseIR::Operand *dataX, const HorseIR::Identifier *dataY)
	{
		const auto& inputOptions = this->m_builder.GetInputOptions();
		const auto parameter = inputOptions.Parameters.at(dataY->GetSymbol());
		const auto shape = inputOptions.ParameterShapes.at(parameter);

		if (const auto listShape = Analysis::ShapeUtils::GetShape<Analysis::ListShape>(shape))
		{
			if (const auto size = Analysis::ShapeUtils::GetSize<Analysis::Shape::ConstantSize>(listShape->GetListSize()))
			{
				for (auto index = 0u; index < size->GetValue(); ++index)
				{
					GenerateInit<T>(dataX, dataY, true, index);
				}
				return;
			}
		}
		Error("non-constant cell count");
	}

	template<class T>
	void GenerateTuple(unsigned int index, const HorseIR::Operand *dataX, const HorseIR::Identifier *dataY)
	{
		if (!this->m_builder.GetInputOptions().IsVectorGeometry())
		{
			Error("tuple-in-list");
		}
		GenerateInit<T>(dataX, dataY, true, index);
	}

private:
	template<class T>
	void GenerateInit(const HorseIR::Operand *dataX, const HorseIR::Identifier *dataY, bool isCell = false, unsigned int cellIndex = 0)
	{
		if constexpr(std::is_same<T, PTX::PredicateType>::value)
		{
			GenerateInit<PTX::Int8Type>(dataX, dataY, isCell, cellIndex);
		}
		else
		{
			// Allocate shared memory space used for data cache

			auto kernelResources = this->m_builder.GetKernelResources();
			
			auto cacheName = dataY->GetName() + "_cache" + ((isCell) ? std::to_string(cellIndex) : "");
			auto s_cache = kernelResources->template AllocateSharedVariable<PTX::ArrayType<T, FIND_BLOCK_SIZE>>(cacheName);

			// Load the X data (cell index default to zero for vector)

			OperandGenerator<B, T> operandGenerator(this->m_builder);
			operandGenerator.GenerateOperand(dataX, OperandGenerator<B, T>::LoadKind::Vector, cellIndex);
		}
	}
};

template<PTX::Bits B>
class InternalFindGenerator_Cache : public Generator
{
public:
	using Generator::Generator; 

	std::string Name() const override { return "InternalFindGenerator_Cache"; }

	void Generate(const HorseIR::Operand *dataX, const HorseIR::Identifier *dataY, const PTX::TypedOperand<PTX::UInt32Type> *globalIndex)
	{
		// Load data into shared memory

		DispatchType(*this, dataX->GetType(), dataY, globalIndex);

		// Synchronize shared memory

		BarrierGenerator<B> barrierGenerator(this->m_builder);
		barrierGenerator.Generate();
	}

	template<class T>
	void GenerateVector(const HorseIR::Identifier *dataY, const PTX::TypedOperand<PTX::UInt32Type> *globalIndex)
	{
		GenerateCache<T>(dataY, globalIndex);
	}

	template<class T>
	void GenerateList(const HorseIR::Identifier *dataY, const PTX::TypedOperand<PTX::UInt32Type> *globalIndex)
	{
		const auto& inputOptions = this->m_builder.GetInputOptions();
		const auto parameter = inputOptions.Parameters.at(dataY->GetSymbol());
		const auto shape = inputOptions.ParameterShapes.at(parameter);

		if (const auto listShape = Analysis::ShapeUtils::GetShape<Analysis::ListShape>(shape))
		{
			if (const auto size = Analysis::ShapeUtils::GetSize<Analysis::Shape::ConstantSize>(listShape->GetListSize()))
			{
				for (auto index = 0u; index < size->GetValue(); ++index)
				{
					GenerateCache<T>(dataY, globalIndex, true, index);
				}
				return;
			}
		}
		Error("non-constant cell count");
	}

	template<class T>
	void GenerateTuple(unsigned int index, const HorseIR::Identifier *dataY, const PTX::TypedOperand<PTX::UInt32Type> *globalIndex)
	{
		if (!this->m_builder.GetInputOptions().IsVectorGeometry())
		{
			Error("tuple-in-list");
		}
		GenerateCache<T>(dataY, globalIndex, true, index);
	}

private:
	template<class T>
	void GenerateCache(const HorseIR::Identifier *dataY, const PTX::TypedOperand<PTX::UInt32Type> *globalIndex, bool isCell = false, unsigned int cellIndex = 0)
	{
		if constexpr(std::is_same<T, PTX::PredicateType>::value)
		{
			GenerateCache<PTX::Int8Type>(dataY, globalIndex, isCell, cellIndex);
		}
		else
		{
			// Get the shared memory cache

			auto kernelResources = this->m_builder.GetKernelResources();

			auto cacheName = dataY->GetName() + "_cache" + ((isCell) ? std::to_string(cellIndex) : "");
			auto s_cache = new PTX::ArrayVariableAdapter<T, FIND_BLOCK_SIZE, PTX::SharedSpace>(
				kernelResources->template GetSharedVariable<PTX::ArrayType<T, FIND_BLOCK_SIZE>>(cacheName)
			);

			// Load the cache from global memory and store in shared (cell index default to zero for vector)

			OperandGenerator<B, T> operandGenerator(this->m_builder);
			auto value = operandGenerator.GenerateRegister(dataY, globalIndex, this->m_builder.UniqueIdentifier("find"), cellIndex);

			ThreadIndexGenerator<B> threadGenerator(this->m_builder);
			auto localIndex = threadGenerator.GenerateLocalIndex();

			AddressGenerator<B, T> addressGenerator(this->m_builder);
			auto s_cacheAddress = addressGenerator.GenerateAddress(s_cache, localIndex);

			this->m_builder.AddStatement(new PTX::StoreInstruction<B, T, PTX::SharedSpace>(s_cacheAddress, value));
		}
	}
};

template<PTX::Bits B>
class InternalFindGenerator_MatchInit : public Generator
{
public:
	using Generator::Generator; 
	using BaseRegistersTy = std::vector<std::pair<const PTX::Register<PTX::UIntType<B>> *, unsigned int>>;

	std::string Name() const override { return "InternalFindGenerator_MatchInit"; }

	BaseRegistersTy Generate(const HorseIR::Operand *dataX, const HorseIR::Identifier *dataY)
	{
		DispatchType(*this, dataX->GetType(), dataY);
		return m_baseRegisters;
	}

	template<class T>
	void GenerateVector(const HorseIR::Identifier *dataY)
	{
		GenerateMatchInit<T>(dataY);
	}

	template<class T>
	void GenerateList(const HorseIR::Identifier *dataY)
	{
		const auto& inputOptions = this->m_builder.GetInputOptions();
		const auto parameter = inputOptions.Parameters.at(dataY->GetSymbol());
		const auto shape = inputOptions.ParameterShapes.at(parameter);

		if (const auto listShape = Analysis::ShapeUtils::GetShape<Analysis::ListShape>(shape))
		{
			if (const auto size = Analysis::ShapeUtils::GetSize<Analysis::Shape::ConstantSize>(listShape->GetListSize()))
			{
				for (auto index = 0u; index < size->GetValue(); ++index)
				{
					GenerateMatchInit<T>(dataY, true, index);
				}
				return;
			}
		}
		Error("non-constant cell count");
	}

	template<class T>
	void GenerateTuple(unsigned int index, const HorseIR::Identifier *dataY)
	{
		if (!this->m_builder.GetInputOptions().IsVectorGeometry())
		{
			Error("tuple-in-list");
		}
		GenerateMatchInit<T>(dataY, true, index);
	}

public:
	template<class T>
	void GenerateMatchInit(const HorseIR::Identifier *dataY, bool isCell = false, unsigned int cellIndex = 0)
	{
		if constexpr(std::is_same<T, PTX::PredicateType>::value)
		{
			GenerateMatchInit<PTX::Int8Type>(dataY, isCell, cellIndex);
		}
		else
		{
			auto resources = this->m_builder.GetLocalResources();
			auto kernelResources = this->m_builder.GetKernelResources();

			// Get cache variable

			auto cacheName = dataY->GetName() + "_cache" + ((isCell) ? std::to_string(cellIndex) : "");
			auto s_cache = new PTX::ArrayVariableAdapter<T, FIND_BLOCK_SIZE, PTX::SharedSpace>(
				kernelResources->template GetSharedVariable<PTX::ArrayType<T, FIND_BLOCK_SIZE>>(cacheName)
			);

			// Load the initial address of the cache (will be increment by the match loop)

			auto base = resources->template AllocateTemporary<PTX::UIntType<B>>();
			auto basePointer = new PTX::PointerRegisterAdapter<B, T, PTX::SharedSpace>(base);
			auto baseAddress = new PTX::MemoryAddress<B, T, PTX::SharedSpace>(s_cache);

			this->m_builder.AddStatement(new PTX::MoveAddressInstruction<B, T, PTX::SharedSpace>(basePointer, baseAddress));

			m_baseRegisters.push_back({base, PTX::BitSize<T::TypeBits>::NumBytes});
		}
	}

	BaseRegistersTy m_baseRegisters;
};

template<PTX::Bits B, class D>
class InternalFindGenerator_Match : public Generator
{
public:
	using BaseRegistersTy = std::vector<std::pair<const PTX::Register<PTX::UIntType<B>> *, unsigned int>>;

	InternalFindGenerator_Match(Builder& builder, const BaseRegistersTy& baseRegisters, FindOperation findOp, const std::vector<ComparisonOperation>& comparisonOps,
			const PTX::Register<PTX::PredicateType> *runningPredicate, const PTX::Register<D>* targetRegister, const PTX::Register<PTX::Int64Type> *writeOffset = nullptr)
		: Generator(builder), m_baseRegisters(baseRegisters), m_findOp(findOp), m_comparisonOps(comparisonOps),
			m_runningPredicate(runningPredicate), m_targetRegister(targetRegister), m_writeOffset(writeOffset) {}

	std::string Name() const override { return "InternalFindGenerator_Match"; }

	void Generate(const HorseIR::Operand *dataX, const HorseIR::Identifier *dataY, unsigned int unrollIndex)
	{
		// Reset matching

		m_matchPredicate = nullptr;

		// Match the cache against the data

		DispatchType(*this, dataX->GetType(), dataX, dataY, unrollIndex);

		// Compute match output result

		if constexpr(std::is_same<D, PTX::PredicateType>::value)
		{
			if (m_findOp == FindOperation::Member)
			{
				this->m_builder.AddStatement(new PTX::OrInstruction<PTX::PredicateType>(m_targetRegister, m_targetRegister, m_matchPredicate));
			}
		}
		else if constexpr(std::is_same<D, PTX::Int64Type>::value)
		{
			switch (m_findOp)
			{
				case FindOperation::Index:
				{
					// Output only the first matching index

					this->m_builder.AddStatement(new PTX::OrInstruction<PTX::PredicateType>(m_runningPredicate, m_runningPredicate, m_matchPredicate));

					auto addInstruction = new PTX::AddInstruction<PTX::Int64Type>(m_targetRegister, m_targetRegister, new PTX::Int64Value(1));
					addInstruction->SetPredicate(m_runningPredicate, true);
					this->m_builder.AddStatement(addInstruction);

					break;
				}
				case FindOperation::Count:
				{
					// Increment the count by 1

					auto addInstruction = new PTX::AddInstruction<PTX::Int64Type>(m_targetRegister, m_targetRegister, new PTX::Int64Value(1));
					addInstruction->SetPredicate(m_matchPredicate);
					this->m_builder.AddStatement(addInstruction);

					break;
				}
				case FindOperation::Indexes:
				{
					// Output all indexes, requires breaking the typical return pattern as there is an indeterminate quantity

					auto resources = this->m_builder.GetLocalResources();
					auto storePredicate = resources->template AllocateTemporary<PTX::PredicateType>();
					auto activePredicate = resources->template AllocateTemporary<PTX::PredicateType>();

					ThreadIndexGenerator<B> indexGenerator(this->m_builder);
					auto globalIndex = indexGenerator.GenerateGlobalIndex();

					DataSizeGenerator<B> sizeGenerator(this->m_builder);
					auto size = sizeGenerator.GenerateSize(dataY);

					this->m_builder.AddStatement(new PTX::SetPredicateInstruction<PTX::UInt32Type>(activePredicate, globalIndex, size, PTX::UInt32Type::ComparisonOperator::Less));
					this->m_builder.AddStatement(new PTX::AndInstruction<PTX::PredicateType>(storePredicate, activePredicate, m_matchPredicate));

					auto label = this->m_builder.CreateLabel("END");
					this->m_builder.AddStatement(new PTX::BranchInstruction(label, storePredicate, true));

					auto kernelResources = this->m_builder.GetKernelResources();
					auto returnParameter = kernelResources->template GetParameter<PTX::PointerType<B, PTX::PointerType<B, PTX::Int64Type, PTX::GlobalSpace>>>(NameUtils::ReturnName(0));

					//TODO: Conversions need to be optimized. Likely by a consistent index size for all data

                                        auto writeOffset = ConversionGenerator::ConvertSource<PTX::UInt32Type, PTX::Int64Type>(this->m_builder, m_writeOffset);

					AddressGenerator<B, PTX::Int64Type> addressGenerator(this->m_builder);
					auto returnAddress0 = addressGenerator.GenerateAddress(returnParameter, 0, writeOffset);
					auto returnAddress1 = addressGenerator.GenerateAddress(returnParameter, 1, writeOffset);

                                        auto globalIndex64 = ConversionGenerator::ConvertSource<PTX::Int64Type, PTX::UInt32Type>(this->m_builder, globalIndex);

					this->m_builder.AddStatement(new PTX::StoreInstruction<B, PTX::Int64Type, PTX::GlobalSpace>(returnAddress0, globalIndex64));
					this->m_builder.AddStatement(new PTX::StoreInstruction<B, PTX::Int64Type, PTX::GlobalSpace>(returnAddress1, m_targetRegister));

					// End control flow and increment both the matched (predicated) and running counts

					this->m_builder.AddStatement(new PTX::AddInstruction<PTX::Int64Type>(m_writeOffset, m_writeOffset, new PTX::Int64Value(1)));
					this->m_builder.AddStatement(label);
					this->m_builder.AddStatement(new PTX::AddInstruction<PTX::Int64Type>(m_targetRegister, m_targetRegister, new PTX::Int64Value(1)));

					break;
				}
			}
		}
	}

	template<class T>
	void GenerateVector(const HorseIR::Operand *dataX, const HorseIR::Identifier *dataY, unsigned int unrollIndex)
	{
		GenerateMatch<T>(dataX, dataY, unrollIndex);
	}

	template<class T>
	void GenerateList(const HorseIR::Operand *dataX, const HorseIR::Identifier *dataY, unsigned int unrollIndex)
	{
		const auto& inputOptions = this->m_builder.GetInputOptions();
		const auto parameter = inputOptions.Parameters.at(dataY->GetSymbol());
		const auto shape = inputOptions.ParameterShapes.at(parameter);

		if (const auto listShape = Analysis::ShapeUtils::GetShape<Analysis::ListShape>(shape))
		{
			if (const auto size = Analysis::ShapeUtils::GetSize<Analysis::Shape::ConstantSize>(listShape->GetListSize()))
			{
				for (auto index = 0u; index < size->GetValue(); ++index)
				{
					GenerateMatch<T>(dataX, dataY, unrollIndex, true, index);
				}
				return;
			}
		}
		Error("non-constant cell count");
	}

	template<class T>
	void GenerateTuple(unsigned int index, const HorseIR::Operand *dataX, const HorseIR::Identifier *dataY, unsigned int unrollIndex)
	{
		if (!this->m_builder.GetInputOptions().IsVectorGeometry())
		{
			Error("tuple-in-list");
		}
		GenerateMatch<T>(dataX, dataY, unrollIndex, true, index);
	}

private:
	template<class T>
	void GenerateMatch(const HorseIR::Operand *dataX, const HorseIR::Identifier *dataY, unsigned int unrollIndex, bool isCell = false, unsigned int cellIndex = 0)
	{
		if constexpr(std::is_same<T, PTX::PredicateType>::value)
		{
			GenerateMatch<PTX::Int8Type>(dataX, dataY, unrollIndex, isCell, cellIndex);
		}
		else
		{
			auto resources = this->m_builder.GetLocalResources();
			auto value = resources->template AllocateTemporary<T>();
			auto predicate = resources->template AllocateTemporary<PTX::PredicateType>();

			// Load data from the cache

			auto base = m_baseRegisters.at(cellIndex).first;
			auto basePointer = new PTX::PointerRegisterAdapter<B, T, PTX::SharedSpace>(base);
			auto s_cacheAddress = new PTX::RegisterAddress<B, T, PTX::SharedSpace>(basePointer, unrollIndex);

			this->m_builder.AddStatement(new PTX::LoadInstruction<B, T, PTX::SharedSpace>(value, s_cacheAddress));

			// Fetch X data (cell index defaults to zero for vector)

			OperandGenerator<B, T> operandGenerator(this->m_builder);
			auto data = operandGenerator.GenerateOperand(dataX, OperandGenerator<B, T>::LoadKind::Vector, cellIndex);

			// Compare cache against the thread data

			auto comparisonOp = m_comparisonOps.at((m_comparisonOps.size() == 1) ? 0 : cellIndex);
			ComparisonGenerator<B, PTX::PredicateType> comparisonGenerator(this->m_builder, comparisonOp);
			comparisonGenerator.Generate(predicate, data, value);

			if (m_matchPredicate == nullptr)
			{
				m_matchPredicate = predicate;
			}
			else
			{
				auto mergedPredicate = resources->template AllocateTemporary<PTX::PredicateType>();
				this->m_builder.AddStatement(new PTX::AndInstruction<PTX::PredicateType>(mergedPredicate, m_matchPredicate, predicate));
				m_matchPredicate = mergedPredicate;
			}
		}
	}

	const PTX::Register<PTX::PredicateType> *m_runningPredicate = nullptr;
	const PTX::Register<PTX::PredicateType> *m_matchPredicate = nullptr;
	const PTX::Register<D> *m_targetRegister = nullptr;

	const PTX::Register<PTX::Int64Type> *m_writeOffset = nullptr;

	BaseRegistersTy m_baseRegisters;

	FindOperation m_findOp;
	std::vector<ComparisonOperation> m_comparisonOps;
};

template<PTX::Bits B, class D>
class InternalFindGenerator : public BuiltinGenerator<B, D>, public HorseIR::ConstVisitor
{
public:
	InternalFindGenerator(Builder& builder, FindOperation findOp, const std::vector<ComparisonOperation>& comparisonOps) : BuiltinGenerator<B, D>(builder), m_findOp(findOp), m_comparisonOps(comparisonOps) {}

	std::string Name() const override { return "InternalFindGenerator"; }

	const PTX::Register<PTX::PredicateType> *GenerateCompressionPredicate(const std::vector<HorseIR::Operand *>& arguments) override
	{
		return OperandCompressionGenerator::UnaryCompressionRegister(this->m_builder, arguments);
	}

	const PTX::Register<D> *Generate(const HorseIR::LValue *target, const std::vector<HorseIR::Operand *>& arguments)
	{
		// Initialize target register (@member->false, @index_of->0, @join_count->0)

		if constexpr(std::is_same<D, PTX::PredicateType>::value)
		{
			if (m_findOp == FindOperation::Member)
			{
				m_targetRegister = this->GenerateTargetRegister(target, arguments);
				this->m_builder.AddStatement(new PTX::MoveInstruction<PTX::PredicateType>(m_targetRegister, new PTX::BoolValue(false)));
			}
		}
		else if constexpr(std::is_same<D, PTX::Int64Type>::value)
		{
			switch (m_findOp)
			{
				case FindOperation::Index:
				{
					auto resources = this->m_builder.GetLocalResources();
					m_runningPredicate = resources->template AllocateTemporary<PTX::PredicateType>();

					this->m_builder.AddStatement(new PTX::MoveInstruction<PTX::PredicateType>(m_runningPredicate, new PTX::BoolValue(false)));
					// Fallthrough
				}
				case FindOperation::Count:
				{
					m_targetRegister = this->GenerateTargetRegister(target, arguments);
					this->m_builder.AddStatement(new PTX::MoveInstruction<PTX::Int64Type>(m_targetRegister, new PTX::Int64Value(0)));

					break;
				}
				case FindOperation::Indexes:
				{
					auto resources = this->m_builder.GetLocalResources();
					m_targetRegister = resources->template AllocateTemporary<PTX::Int64Type>();

					this->m_builder.AddStatement(new PTX::MoveInstruction<PTX::Int64Type>(m_targetRegister, new PTX::Int64Value(0)));

					OperandGenerator<B, PTX::Int64Type> operandGenerator(this->m_builder);
					m_writeOffset = operandGenerator.GenerateRegister(arguments.at(2), OperandGenerator<B, PTX::Int64Type>::LoadKind::Vector);//TODO: Unique load; this->m_builder.UniqueIdentifier("find"));

					break;
				}
			}
		}

		// Operating range in argument 0, possibilities in argument 1

		m_dataX = arguments.at(0);
		arguments.at(1)->Accept(*this);

		return m_targetRegister;
	}

	void Visit(const HorseIR::Identifier *identifierY) override
	{
		// Generate a double loop, checking for matches in chunks (for coalescing)
		//
		// __shared__ T s[FIND_BLOCK_SIZE]
		//
		// found = false
		//
		// i = threadIdx.x
		// START_1:
		//    setp %p1, i, identifier.size
		//    @%p1 br END_1
		//
		//    s[threadIdx.x] = value[i]
		//
		//    <barrier>
		//
		//    <inner loop>
		//
		//    <barrier>
		//    <increment, i, FIND_BLOCK_SIZE>
		//
		//    br START_1
		//
		// END_1:

		auto resources = this->m_builder.GetLocalResources();

		// Maximize the block size

		auto& kernelOptions = this->m_builder.GetKernelOptions();
		kernelOptions.SetBlockSize(FIND_BLOCK_SIZE);

		ThreadIndexGenerator<B> threadGenerator(this->m_builder);
		DataSizeGenerator<B> sizeGenerator(this->m_builder);

		// Initialize data cache and X0 vector

		InternalFindGenerator_Init<B> initGenerator(this->m_builder);
		initGenerator.Generate(m_dataX, identifierY);

		// Initialize the outer loop

		auto size = sizeGenerator.GenerateSize(identifierY);
		auto index = threadGenerator.GenerateLocalIndex();

		auto bound = resources->template AllocateTemporary<PTX::UInt32Type>();
		this->m_builder.AddStatement(new PTX::AddInstruction<PTX::UInt32Type>(bound, index, size));

		// Check if the final iteration is complete

		auto remainder = resources->template AllocateTemporary<PTX::UInt32Type>();
		this->m_builder.AddStatement(new PTX::RemainderInstruction<PTX::UInt32Type>(remainder, size, new PTX::UInt32Value(FIND_BLOCK_SIZE)));

		auto sizePredicate_1 = resources->template AllocateTemporary<PTX::PredicateType>();
		this->m_builder.AddStatement(
			new PTX::SetPredicateInstruction<PTX::UInt32Type>(sizePredicate_1, remainder, new PTX::UInt32Value(0), PTX::UInt32Type::ComparisonOperator::NotEqual)
		);

		auto startLabel = this->m_builder.CreateLabel("START");
		auto endLabel = this->m_builder.CreateLabel("END");
		auto predicate = resources->template AllocateTemporary<PTX::PredicateType>();

		this->m_builder.AddStatement(startLabel);
		this->m_builder.AddStatement(new PTX::SetPredicateInstruction<PTX::UInt32Type>(predicate, index, bound, PTX::UInt32Type::ComparisonOperator::GreaterEqual));
		this->m_builder.AddStatement(new PTX::BranchInstruction(endLabel, predicate));
		this->m_builder.AddStatement(new PTX::BlankStatement());

		// Load the data cache into shared memory and synchronize

		InternalFindGenerator_Cache<B> cacheGenerator(this->m_builder);
		cacheGenerator.Generate(m_dataX, identifierY, index);

		// Setup the inner loop and bound
		//  - Chunks 0...(N-1): FIND_BLOCK_SIZE
		//  - Chunk N: remainder

		// Increment by the number of threads. Do it here since it will be reused

		this->m_builder.AddStatement(new PTX::AddInstruction<PTX::UInt32Type>(index, index, new PTX::UInt32Value(FIND_BLOCK_SIZE)));

		// Chose the correct bound for the inner loop, sizePredicate indicates a complete iteration

		auto sizePredicate_2 = resources->template AllocateTemporary<PTX::PredicateType>();
		auto sizePredicate_3 = resources->template AllocateTemporary<PTX::PredicateType>();

		this->m_builder.AddStatement(new PTX::SetPredicateInstruction<PTX::UInt32Type>(sizePredicate_2, index, bound, PTX::UInt32Type::ComparisonOperator::GreaterEqual));
		this->m_builder.AddStatement(new PTX::AndInstruction<PTX::PredicateType>(sizePredicate_3, sizePredicate_1, sizePredicate_2));

		auto ifElseLabel = this->m_builder.CreateLabel("ELSE"); 
		auto ifEndLabel = this->m_builder.CreateLabel("END"); 

		this->m_builder.AddStatement(new PTX::BranchInstruction(ifElseLabel, sizePredicate_3));
		this->m_builder.AddStatement(new PTX::BlankStatement());

		GenerateInnerLoop(identifierY, new PTX::UInt32Value(FIND_BLOCK_SIZE), 16);

		this->m_builder.AddStatement(new PTX::BranchInstruction(ifEndLabel));
		this->m_builder.AddStatement(new PTX::BlankStatement());
		this->m_builder.AddStatement(ifElseLabel);

		// Compute bound if this is the last iteration

		GenerateInnerLoop(identifierY, remainder, 1);
		
		this->m_builder.AddStatement(ifEndLabel);

		// Barrier before the next chunk, otherwise some warps might overwrite the previous values too soon

		BarrierGenerator<B> barrierGenerator(this->m_builder);
		barrierGenerator.Generate();

		// Complete the loop structure, check the next index chunk

		this->m_builder.AddStatement(new PTX::BranchInstruction(startLabel));
		this->m_builder.AddStatement(new PTX::BlankStatement());
		this->m_builder.AddStatement(endLabel);
	}

	void GenerateInnerLoop(const HorseIR::Identifier *identifierY, const PTX::TypedOperand<PTX::UInt32Type> *bound, unsigned int factor)
	{
		auto resources = this->m_builder.GetLocalResources();

		// Loop body
		//
		//    j = 0
		//    START_2: (unroll)
		//       setp %p2, j, FIND_BLOCK_SIZE
		//       @%p2 br END_2
		//
		//       <check= s[j]> (set found, do NOT break -> slower)
		//
		//       <increment, j, 1>
		//       br START_2
		//
		//    END_2:

		auto index = resources->template AllocateTemporary<PTX::UInt32Type>();
		this->m_builder.AddStatement(new PTX::MoveInstruction<PTX::UInt32Type>(index, new PTX::UInt32Value(0)));

		InternalFindGenerator_MatchInit<B> initGenerator(this->m_builder);
		auto baseRegisters = initGenerator.Generate(m_dataX, identifierY);

		auto startLabel = this->m_builder.CreateLabel("START");
		auto endLabel = this->m_builder.CreateLabel("END");
		auto predicate = resources->template AllocateTemporary<PTX::PredicateType>();

		this->m_builder.AddStatement(startLabel);
		this->m_builder.AddStatement(new PTX::SetPredicateInstruction<PTX::UInt32Type>(predicate, index, bound, PTX::UInt32Type::ComparisonOperator::GreaterEqual));
		this->m_builder.AddStatement(new PTX::BranchInstruction(endLabel, predicate));
		this->m_builder.AddStatement(new PTX::BlankStatement());

		// Generate match for every unroll factor

		InternalFindGenerator_Match<B, D> matchGenerator(this->m_builder, baseRegisters, m_findOp, m_comparisonOps, m_runningPredicate, m_targetRegister, m_writeOffset);
		for (auto i = 0; i < factor; ++i)
		{
			matchGenerator.Generate(m_dataX, identifierY, i);
		}

		// Increment by the iterations completed

		this->m_builder.AddStatement(new PTX::AddInstruction<PTX::UInt32Type>(index, index, new PTX::UInt32Value(factor)));

		for (const auto [base, inc] : baseRegisters)
		{
			this->m_builder.AddStatement(new PTX::AddInstruction<PTX::UIntType<B>>(base, base, new PTX::UIntValue<B>(factor * inc)));
		}

		this->m_builder.AddStatement(new PTX::BranchInstruction(startLabel));
		this->m_builder.AddStatement(new PTX::BlankStatement());
		this->m_builder.AddStatement(endLabel);     
	}

	void Visit(const HorseIR::Literal *literal) override
	{
		BuiltinGenerator<B, D>::Unimplemented("literal kind");
	}

	void Visit(const HorseIR::BooleanLiteral *literal) override
	{
		VisitLiteral<std::int8_t>(literal);
	}

	void Visit(const HorseIR::CharLiteral *literal) override
	{
		VisitLiteral<std::int8_t>(literal);
	}

	void Visit(const HorseIR::Int8Literal *literal) override
	{
		VisitLiteral<std::int8_t>(literal);
	}

	void Visit(const HorseIR::Int16Literal *literal) override
	{
		VisitLiteral<std::int16_t>(literal);
	}

	void Visit(const HorseIR::Int32Literal *literal) override
	{
		VisitLiteral<std::int32_t>(literal);
	}

	void Visit(const HorseIR::Int64Literal *literal) override
	{
		VisitLiteral<std::int64_t>(literal);
	}

	void Visit(const HorseIR::Float32Literal *literal) override
	{
		VisitLiteral<float>(literal);
	}

	void Visit(const HorseIR::Float64Literal *literal) override
	{
		VisitLiteral<double>(literal);
	}

	void Visit(const HorseIR::StringLiteral *literal) override
	{
		VisitLiteral<std::string>(literal);
	}

	void Visit(const HorseIR::SymbolLiteral *literal) override
	{
		VisitLiteral<HorseIR::SymbolValue *>(literal);
	}

	void Visit(const HorseIR::DatetimeLiteral *literal) override
	{
		VisitLiteral<HorseIR::DatetimeValue *>(literal);
	}

	void Visit(const HorseIR::MonthLiteral *literal) override
	{
		VisitLiteral<HorseIR::MonthValue *>(literal);
	}

	void Visit(const HorseIR::DateLiteral *literal) override
	{
		VisitLiteral<HorseIR::DateValue *>(literal);
	}

	void Visit(const HorseIR::MinuteLiteral *literal) override
	{
		VisitLiteral<HorseIR::MinuteValue *>(literal);
	}

	void Visit(const HorseIR::SecondLiteral *literal) override
	{
		VisitLiteral<HorseIR::SecondValue *>(literal);
	}

	void Visit(const HorseIR::TimeLiteral *literal) override
	{
		VisitLiteral<HorseIR::TimeValue *>(literal);
	}

	template<class L>
	void VisitLiteral(const HorseIR::TypedVectorLiteral<L> *literal)
	{
		// For each value in the literal, check if it is equal to the data value in this thread. Note, this is an unrolled loop

		//TODO: Constant match
		// for (const auto& value : literal->GetValues())
		// {
		// 	// Load the value and cast to the appropriate type

		// 	if constexpr(std::is_same<L, std::string>::value)
		// 	{
		// 		GenerateMatch(new PTX::Value<T>(static_cast<typename T::SystemType>(Runtime::StringBucket::HashString(value))));
		// 	}
		// 	else if constexpr(std::is_same<L, HorseIR::SymbolValue *>::value)
		// 	{
		// 		GenerateMatch(new PTX::Value<T>(static_cast<typename T::SystemType>(Runtime::StringBucket::HashString(value->GetName()))));
		// 	}
		// 	else if constexpr(std::is_convertible<L, HorseIR::CalendarValue *>::value)
		// 	{
		// 		GenerateMatch(new PTX::Value<T>(static_cast<typename T::SystemType>(value->GetEpochTime())));
		// 	}
		// 	else if constexpr(std::is_convertible<L, HorseIR::ExtendedCalendarValue *>::value)
		// 	{
		// 		GenerateMatch(new PTX::Value<T>(static_cast<typename T::SystemType>(value->GetExtendedEpochTime())));
		// 	}
		// 	else
		// 	{
		// 		GenerateMatch(new PTX::Value<T>(static_cast<typename T::SystemType>(value)));
		// 	}
		// }
	}

private:
	const PTX::Register<PTX::PredicateType> *m_runningPredicate = nullptr;
	const PTX::Register<D> *m_targetRegister = nullptr;

	const PTX::Register<PTX::Int64Type> *m_writeOffset = nullptr; // Join output

	const HorseIR::Operand *m_dataX = nullptr;

	FindOperation m_findOp;
	std::vector<ComparisonOperation> m_comparisonOps;
};

}
