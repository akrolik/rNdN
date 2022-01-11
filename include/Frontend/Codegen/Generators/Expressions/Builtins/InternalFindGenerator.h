#pragma once

#include <utility>

#include "HorseIR/Traversal/ConstVisitor.h"
#include "Frontend/Codegen/Generators/Expressions/Builtins/BuiltinGenerator.h"

#include "Frontend/Codegen/Builder.h"
#include "Frontend/Codegen/Generators/Expressions/ConversionGenerator.h"
#include "Frontend/Codegen/Generators/Expressions/OperandCompressionGenerator.h"
#include "Frontend/Codegen/Generators/Expressions/OperandGenerator.h"
#include "Frontend/Codegen/Generators/Expressions/Builtins/ComparisonGenerator.h"
#include "Frontend/Codegen/Generators/Expressions/Builtins/InternalCacheGenerator.h"
#include "Frontend/Codegen/Generators/Indexing/AddressGenerator.h"
#include "Frontend/Codegen/Generators/Indexing/DataSizeGenerator.h"
#include "Frontend/Codegen/Generators/Indexing/ThreadIndexGenerator.h"
#include "Frontend/Codegen/Generators/Synchronization/BarrierGenerator.h"
#include "Frontend/Codegen/NameUtils.h"

#include "HorseIR/Analysis/Shape/Shape.h"
#include "HorseIR/Analysis/Shape/ShapeUtils.h"

#include "HorseIR/Tree/Tree.h"
#include "HorseIR/Utils/TypeUtils.h"

#include "PTX/Tree/Tree.h"

namespace Frontend {
namespace Codegen {

enum class FindOperation {
	Member,
	Index,
	Count,
	Indexes
};

constexpr auto FIND_CACHE_SIZE = 1024u;

template<PTX::Bits B>
class InternalFindGenerator_Init : public Generator
{
public:
	using Generator::Generator; 

	std::string Name() const override { return "InternalFindGenerator_Init"; }

	void Generate(const HorseIR::Operand *dataX, const HorseIR::Analysis::Shape *shape)
	{
		DispatchType(*this, dataX->GetType(), dataX, shape);
	}

	template<class T>
	void GenerateVector(const HorseIR::Operand *dataX, const HorseIR::Analysis::Shape *shape)
	{
		GenerateInit<T>(dataX);
	}

	template<class T>
	void GenerateList(const HorseIR::Operand *dataX, const HorseIR::Analysis::Shape *shape)
	{
		if (const auto listShape = HorseIR::Analysis::ShapeUtils::GetShape<HorseIR::Analysis::ListShape>(shape))
		{
			if (const auto size = HorseIR::Analysis::ShapeUtils::GetSize<HorseIR::Analysis::Shape::ConstantSize>(listShape->GetListSize()))
			{
				for (auto index = 0u; index < size->GetValue(); ++index)
				{
					GenerateInit<T>(dataX, index);
				}
				return;
			}
		}
		Error("non-constant cell count");
	}

	template<class T>
	void GenerateTuple(unsigned int index, const HorseIR::Operand *dataX, const HorseIR::Analysis::Shape *shape)
	{
		if (!this->m_builder.GetInputOptions().IsVectorGeometry())
		{
			Error("tuple-in-list");
		}
		GenerateInit<T>(dataX, index);
	}

private:
	template<class T>
	void GenerateInit(const HorseIR::Operand *dataX, unsigned int cellIndex = 0)
	{
		if constexpr(std::is_same<T, PTX::PredicateType>::value)
		{
			GenerateInit<PTX::Int8Type>(dataX, cellIndex);
		}
		else
		{
			// Load the X data (cell index default to zero for vector)

			OperandGenerator<B, T> operandGenerator(this->m_builder);
			operandGenerator.GenerateOperand(dataX, OperandGenerator<B, T>::LoadKind::Vector, cellIndex);
		}
	}
};

template<PTX::Bits B>
class InternalFindGenerator_MatchInit : public Generator
{
public:
	using Generator::Generator; 
	using BaseRegistersTy = std::vector<std::pair<PTX::Register<PTX::UIntType<B>> *, unsigned int>>;

	std::string Name() const override { return "InternalFindGenerator_MatchInit"; }

	BaseRegistersTy Generate(const HorseIR::Identifier *dataY, const HorseIR::Type *type)
	{
		DispatchType(*this, type, dataY);
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

		if (const auto listShape = HorseIR::Analysis::ShapeUtils::GetShape<HorseIR::Analysis::ListShape>(shape))
		{
			if (const auto size = HorseIR::Analysis::ShapeUtils::GetSize<HorseIR::Analysis::Shape::ConstantSize>(listShape->GetListSize()))
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
			auto s_cache = new PTX::ArrayVariableAdapter<T, FIND_CACHE_SIZE, PTX::SharedSpace>(
				kernelResources->template GetSharedVariable<PTX::ArrayType<T, FIND_CACHE_SIZE>>(cacheName)
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
class InternalFindGenerator_Update : public Generator
{
public:
	InternalFindGenerator_Update(
		Builder& builder, FindOperation findOp,
		PTX::Register<PTX::PredicateType> *runningPredicate, PTX::Register<D>* targetRegister, PTX::Register<PTX::UInt32Type> *writeOffset = nullptr
	) : Generator(builder), m_findOp(findOp), m_runningPredicate(runningPredicate), m_targetRegister(targetRegister), m_writeOffset(writeOffset) {}

	std::string Name() const override { return "InternalFindGenerator_Update"; }

	void Generate(const HorseIR::Operand *dataX, PTX::Register<PTX::PredicateType> *matchPredicate)
	{
		// Compute match output result

		if constexpr(std::is_same<D, PTX::PredicateType>::value)
		{
			if (m_findOp == FindOperation::Member)
			{
				this->m_builder.AddStatement(new PTX::OrInstruction<PTX::PredicateType>(m_targetRegister, m_targetRegister, matchPredicate));
			}
		}
		else if constexpr(std::is_same<D, PTX::Int64Type>::value)
		{
			switch (m_findOp)
			{
				case FindOperation::Index:
				{
					// Output only the first matching index

					this->m_builder.AddStatement(new PTX::OrInstruction<PTX::PredicateType>(m_runningPredicate, m_runningPredicate, matchPredicate));

					auto addInstruction = new PTX::AddInstruction<PTX::Int64Type>(m_targetRegister, m_targetRegister, new PTX::Int64Value(1));
					addInstruction->SetPredicate(m_runningPredicate, true);
					this->m_builder.AddStatement(addInstruction);

					break;
				}
				case FindOperation::Count:
				{
					// Increment the count by 1

					auto addInstruction = new PTX::AddInstruction<PTX::Int64Type>(m_targetRegister, m_targetRegister, new PTX::Int64Value(1));
					addInstruction->SetPredicate(matchPredicate);
					this->m_builder.AddStatement(addInstruction);

					break;
				}
				case FindOperation::Indexes:
				{
					auto resources = this->m_builder.GetLocalResources();

					// Output all indexes, requires breaking the typical return pattern as there is an indeterminate quantity

					DataIndexGenerator<B> indexGenerator(this->m_builder);
					auto index = indexGenerator.GenerateDataIndex();

					DataSizeGenerator<B> sizeGenerator(this->m_builder);
					auto size = sizeGenerator.GenerateSize(dataX);

					this->m_builder.AddIfStatement("FIND", [&]()
					{
						auto storePredicate = resources->template AllocateTemporary<PTX::PredicateType>();
						auto activePredicate = resources->template AllocateTemporary<PTX::PredicateType>();

						this->m_builder.AddStatement(new PTX::SetPredicateInstruction<PTX::UInt32Type>(
							activePredicate, index, size, PTX::UInt32Type::ComparisonOperator::Less
						));
						this->m_builder.AddStatement(new PTX::AndInstruction<PTX::PredicateType>(
							storePredicate, activePredicate, matchPredicate
						));

						return std::make_tuple(storePredicate, true);
					},
					[&]()
					{
						auto kernelResources = this->m_builder.GetKernelResources();
						auto returnParameter = kernelResources->template GetParameter<PTX::PointerType<B, PTX::PointerType<B, PTX::Int64Type, PTX::GlobalSpace>>>(NameUtils::ReturnName(0));

						AddressGenerator<B, PTX::Int64Type> addressGenerator(this->m_builder);
						auto returnAddress0 = addressGenerator.GenerateAddress(returnParameter, 0, m_writeOffset);
						auto returnAddress1 = addressGenerator.GenerateAddress(returnParameter, 1, m_writeOffset);

						auto index64 = ConversionGenerator::ConvertSource<PTX::Int64Type, PTX::UInt32Type>(this->m_builder, index);

						this->m_builder.AddStatement(new PTX::StoreInstruction<B, PTX::Int64Type, PTX::GlobalSpace>(returnAddress0, m_targetRegister));
						this->m_builder.AddStatement(new PTX::StoreInstruction<B, PTX::Int64Type, PTX::GlobalSpace>(returnAddress1, index64));

						// End control flow and increment both the matched (predicated) and running counts

						this->m_builder.AddStatement(new PTX::AddInstruction<PTX::UInt32Type>(m_writeOffset, m_writeOffset, new PTX::UInt32Value(1)));
					});

					this->m_builder.AddStatement(new PTX::AddInstruction<PTX::Int64Type>(m_targetRegister, m_targetRegister, new PTX::Int64Value(1)));

					break;
				}
			}
		}
	}

private:
	PTX::Register<PTX::PredicateType> *m_runningPredicate = nullptr;
	PTX::Register<D> *m_targetRegister = nullptr;
	PTX::Register<PTX::UInt32Type> *m_writeOffset = nullptr;

	FindOperation m_findOp;
};

template<PTX::Bits B, class D>
class InternalFindGenerator_Match : public Generator
{
public:
	using BaseRegistersTy = std::vector<std::pair<PTX::Register<PTX::UIntType<B>> *, unsigned int>>;

	InternalFindGenerator_Match(
		Builder& builder, const BaseRegistersTy& baseRegisters, FindOperation findOp, const std::vector<ComparisonOperation>& comparisonOps,
		PTX::Register<PTX::PredicateType> *runningPredicate, PTX::Register<D>* targetRegister, PTX::Register<PTX::UInt32Type> *writeOffset = nullptr
	) : Generator(builder), m_baseRegisters(baseRegisters), m_findOp(findOp), m_comparisonOps(comparisonOps),
		m_runningPredicate(runningPredicate), m_targetRegister(targetRegister), m_writeOffset(writeOffset) {}

	std::string Name() const override { return "InternalFindGenerator_Match"; }

	void Generate(const HorseIR::Operand *dataX, const HorseIR::Identifier *dataY, unsigned int unrollIndex)
	{
		// Reset matching

		m_matchPredicate = nullptr;

		// Match the cache against the data

		DispatchType(*this, dataX->GetType(), dataX, dataY, unrollIndex);

		InternalFindGenerator_Update<B, D> updateGenerator(this->m_builder, m_findOp, m_runningPredicate, m_targetRegister, m_writeOffset);
		updateGenerator.Generate(dataX, m_matchPredicate);
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

		if (const auto listShape = HorseIR::Analysis::ShapeUtils::GetShape<HorseIR::Analysis::ListShape>(shape))
		{
			if (const auto size = HorseIR::Analysis::ShapeUtils::GetSize<HorseIR::Analysis::Shape::ConstantSize>(listShape->GetListSize()))
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

	PTX::Register<PTX::PredicateType> *m_runningPredicate = nullptr;
	PTX::Register<PTX::PredicateType> *m_matchPredicate = nullptr;
	PTX::Register<D> *m_targetRegister = nullptr;

	PTX::Register<PTX::UInt32Type> *m_writeOffset = nullptr;

	BaseRegistersTy m_baseRegisters;

	FindOperation m_findOp;
	std::vector<ComparisonOperation> m_comparisonOps;
};

template<PTX::Bits B, class D, typename L>
class InternalFindGenerator_Constant : public Generator
{
public:
	InternalFindGenerator_Constant(
		Builder& builder, FindOperation findOp, ComparisonOperation comparisonOp,
		PTX::Register<PTX::PredicateType> *runningPredicate, PTX::Register<D>* targetRegister, PTX::Register<PTX::UInt32Type> *writeOffset = nullptr
	) : Generator(builder), m_findOp(findOp), m_comparisonOp(comparisonOp),
		m_runningPredicate(runningPredicate), m_targetRegister(targetRegister), m_writeOffset(writeOffset) {}

	std::string Name() const override { return "InternalFindGenerator_Constant"; }

	void Generate(const HorseIR::Operand *dataX, const HorseIR::TypedVectorLiteral<L> *dataY)
	{
		auto widestType = HorseIR::TypeUtils::WidestType(dataX->GetType(), dataY->GetType());
		DispatchType(*this, widestType, dataX, dataY);
	}

	template<class T>
	void GenerateVector(const HorseIR::Operand *dataX, const HorseIR::TypedVectorLiteral<L> *dataY)
	{
		// For each value in the literal, check if it is equal to the data value in this thread. Note, this is an unrolled loop

		OperandGenerator<B, T> operandGenerator(this->m_builder);
		auto valueX = operandGenerator.GenerateOperand(dataX, OperandGenerator<B, T>::LoadKind::Vector);

		for (const auto& valueY : dataY->GetValues())
		{
			// Load the value and cast to the appropriate type

			if constexpr(std::is_same<L, std::string>::value)
			{
				GenerateMatch(dataX, valueX, new PTX::Value<T>(static_cast<typename T::SystemType>(Runtime::StringBucket::HashString(valueY))));
			}
			else if constexpr(std::is_same<L, HorseIR::SymbolValue *>::value)
			{
				GenerateMatch(dataX, valueX, new PTX::Value<T>(static_cast<typename T::SystemType>(Runtime::StringBucket::HashString(valueY->GetName()))));
			}
			else if constexpr(std::is_convertible<L, HorseIR::CalendarValue *>::value)
			{
				GenerateMatch(dataX, valueX, new PTX::Value<T>(static_cast<typename T::SystemType>(valueY->GetEpochTime())));
			}
			else if constexpr(std::is_convertible<L, HorseIR::ExtendedCalendarValue *>::value)
			{
				GenerateMatch(dataX, valueX, new PTX::Value<T>(static_cast<typename T::SystemType>(valueY->GetExtendedEpochTime())));
			}
			else
			{
				GenerateMatch(dataX, valueX, new PTX::Value<T>(static_cast<typename T::SystemType>(valueY)));
			}
		}
	}

	template<class T>
	void GenerateList(const HorseIR::Operand *dataX, const HorseIR::TypedVectorLiteral<L> *dataY)
	{
		Error("literal find for list type");
	}

	template<class T>
	void GenerateTuple(unsigned int index, const HorseIR::Operand *dataX, const HorseIR::TypedVectorLiteral<L> *dataY)
	{
		Error("literal find for tuple type");
	}

private:
	template<class T>
	void GenerateMatch(const HorseIR::Operand *dataX, PTX::TypedOperand<T> *valueX, PTX::TypedOperand<T> *valueY)
	{
		auto resources = this->m_builder.GetLocalResources();
		auto matchPredicate = resources->template AllocateTemporary<PTX::PredicateType>();

		ComparisonGenerator<B, PTX::PredicateType> comparisonGenerator(this->m_builder, m_comparisonOp);
		comparisonGenerator.Generate(matchPredicate, valueX, valueY);

		InternalFindGenerator_Update<B, D> updateGenerator(this->m_builder, m_findOp, m_runningPredicate, m_targetRegister, m_writeOffset);
		updateGenerator.Generate(dataX, matchPredicate);
	}

	PTX::Register<PTX::PredicateType> *m_runningPredicate = nullptr;
	PTX::Register<D> *m_targetRegister = nullptr;

	PTX::Register<PTX::UInt32Type> *m_writeOffset = nullptr;

	FindOperation m_findOp;
	ComparisonOperation m_comparisonOp;
};

template<PTX::Bits B, class D>
class InternalFindGenerator : public BuiltinGenerator<B, D>, public HorseIR::ConstVisitor
{
public:
	InternalFindGenerator(Builder& builder, FindOperation findOp, const std::vector<ComparisonOperation>& comparisonOps) : BuiltinGenerator<B, D>(builder), m_findOp(findOp), m_comparisonOps(comparisonOps) {}

	std::string Name() const override { return "InternalFindGenerator"; }

	PTX::Register<PTX::PredicateType> *GenerateCompressionPredicate(const std::vector<const HorseIR::Operand *>& arguments) override
	{
		return OperandCompressionGenerator::UnaryCompressionRegister(this->m_builder, arguments);
	}

	PTX::Register<D> *Generate(const HorseIR::LValue *target, const std::vector<const HorseIR::Operand *>& arguments)
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
					m_writeOffset = resources->template AllocateTemporary<PTX::UInt32Type>();

					this->m_builder.AddStatement(new PTX::MoveInstruction<PTX::Int64Type>(m_targetRegister, new PTX::Int64Value(0)));

					OperandGenerator<B, PTX::UInt32Type> operandGenerator(this->m_builder);
					auto writeOffset = operandGenerator.GenerateOperand(arguments.at(2), OperandGenerator<B, PTX::UInt32Type>::LoadKind::Vector);
					this->m_builder.AddStatement(new PTX::MoveInstruction<PTX::UInt32Type>(m_writeOffset, writeOffset));

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
		// __shared__ T s[FIND_CACHE_SIZE]
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
		//    <increment, i, FIND_CACHE_SIZE>
		//
		//    br START_1
		//
		// END_1:

		auto resources = this->m_builder.GetLocalResources();

		// Maximize the block size

		this->m_builder.SetBlockSize(FIND_CACHE_SIZE);

		ThreadIndexGenerator<B> threadGenerator(this->m_builder);
		DataSizeGenerator<B> sizeGenerator(this->m_builder);

		// Initialize X0 vector

		auto& inputOptions = this->m_builder.GetInputOptions();
		auto parameterY = inputOptions.Parameters.at(identifierY->GetSymbol());
		auto shapeY = inputOptions.ParameterShapes.at(parameterY);

		InternalFindGenerator_Init<B> initGenerator(this->m_builder);
		initGenerator.Generate(m_dataX, shapeY);

		// Initialize the outer loop

		auto size = sizeGenerator.GenerateSize(identifierY);
		auto index = threadGenerator.GenerateLocalIndex();

		auto bound = resources->template AllocateTemporary<PTX::UInt32Type>();
		this->m_builder.AddStatement(new PTX::AddInstruction<PTX::UInt32Type>(bound, index, size));

		// Check if the final iteration is complete

		auto remainder = resources->template AllocateTemporary<PTX::UInt32Type>();
		this->m_builder.AddStatement(new PTX::RemainderInstruction<PTX::UInt32Type>(remainder, size, new PTX::UInt32Value(FIND_CACHE_SIZE)));

		this->m_builder.AddIfStatement("FIND_SKIP", [&]()
		{
			auto predicate = resources->template AllocateTemporary<PTX::PredicateType>();
			this->m_builder.AddStatement(new PTX::SetPredicateInstruction<PTX::UInt32Type>(
				predicate, index, bound, PTX::UInt32Type::ComparisonOperator::GreaterEqual
			));
			return std::make_tuple(predicate, false);
		},
		[&]()
		{
			this->m_builder.AddDoWhileLoop("FIND", [&](Builder::LoopContext& loopContext)
			{
				// Load the data cache into shared memory and synchronize

				InternalCacheGenerator_Load<B, FIND_CACHE_SIZE, 1> cacheGenerator(this->m_builder);
				cacheGenerator.Generate(identifierY, index, m_dataX->GetType());

				// Setup the inner loop and bound
				//  - Chunks 0...(N-1): FIND_CACHE_SIZE
				//  - Chunk N: remainder

				// Increment by the number of threads. Do it here since it will be reused

				this->m_builder.AddStatement(new PTX::AddInstruction<PTX::UInt32Type>(index, index, new PTX::UInt32Value(FIND_CACHE_SIZE)));

				this->m_builder.AddIfElseStatement("FINDA", [&]()
				{
					// Chose the correct bound for the inner loop, sizePredicate indicates a complete iteration

					auto sizePredicate_1 = resources->template AllocateTemporary<PTX::PredicateType>();
					auto sizePredicate_2 = resources->template AllocateTemporary<PTX::PredicateType>();
					auto sizePredicate_3 = resources->template AllocateTemporary<PTX::PredicateType>();

					this->m_builder.AddStatement(new PTX::SetPredicateInstruction<PTX::UInt32Type>(
						sizePredicate_1, remainder, new PTX::UInt32Value(0), PTX::UInt32Type::ComparisonOperator::NotEqual
					));
					this->m_builder.AddStatement(new PTX::SetPredicateInstruction<PTX::UInt32Type>(
						sizePredicate_2, index, bound, PTX::UInt32Type::ComparisonOperator::GreaterEqual
					));
					this->m_builder.AddStatement(new PTX::AndInstruction<PTX::PredicateType>(sizePredicate_3, sizePredicate_1, sizePredicate_2));

					return std::make_tuple(sizePredicate_3, false);
				},
				[&]()
				{
					GenerateInnerLoop(identifierY, new PTX::UInt32Value(FIND_CACHE_SIZE), 16);
				},
				[&]()
				{
					GenerateInnerLoop(identifierY, remainder, 1);
				});

				// Barrier before the next chunk, otherwise some warps might overwrite the previous values too soon

				BarrierGenerator<B> barrierGenerator(this->m_builder);
				barrierGenerator.Generate();

				// Exit loop if last chunk

				auto predicate = resources->template AllocateTemporary<PTX::PredicateType>();
				this->m_builder.AddStatement(new PTX::SetPredicateInstruction<PTX::UInt32Type>(
					predicate, index, bound, PTX::UInt32Type::ComparisonOperator::Less
				));
				return std::make_tuple(predicate, false);
			});
		});
	}

	void GenerateInnerLoop(const HorseIR::Identifier *identifierY, PTX::TypedOperand<PTX::UInt32Type> *bound, unsigned int factor)
	{
		auto resources = this->m_builder.GetLocalResources();

		// Loop body
		//
		//    j = 0
		//
		//    setp.ge %p2, j, FIND_CACHE_SIZE
		//    @%p2 br END_2
		//
		//    START_2: (unroll)
		//       <check= s[j]> (set found, do NOT break -> slower)
		//
		//       <increment, j, unroll>
		//
		//       setp.lt %p2, j, FIND_CACHE_SIZE
		//       @%p2 br START_2
		//
		//    END_2:

		auto index = resources->template AllocateTemporary<PTX::UInt32Type>();
		this->m_builder.AddStatement(new PTX::MoveInstruction<PTX::UInt32Type>(index, new PTX::UInt32Value(0)));

		InternalFindGenerator_MatchInit<B> initGenerator(this->m_builder);
		auto baseRegisters = initGenerator.Generate(identifierY, m_dataX->GetType());

		this->m_builder.AddIfStatement("FINDI_SKIP", [&]()
		{
			auto predicate = resources->template AllocateTemporary<PTX::PredicateType>();
			this->m_builder.AddStatement(new PTX::SetPredicateInstruction<PTX::UInt32Type>(
				predicate, index, bound, PTX::UInt32Type::ComparisonOperator::GreaterEqual
			));
			return std::make_tuple(predicate, false);
		},
		[&]()
		{
			this->m_builder.AddDoWhileLoop("FINDI", [&](Builder::LoopContext& loopContext)
			{
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
				
				// Exit loop

				auto predicate = resources->template AllocateTemporary<PTX::PredicateType>();
				this->m_builder.AddStatement(new PTX::SetPredicateInstruction<PTX::UInt32Type>(
					predicate, index, bound, PTX::UInt32Type::ComparisonOperator::Less
				));
				return std::make_tuple(predicate, false);
			});
		});
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
		InternalFindGenerator_Constant<B, D, L> constantGenerator(
			this->m_builder, m_findOp, m_comparisonOps.at(0), m_runningPredicate, m_targetRegister, m_writeOffset
		);
		constantGenerator.Generate(m_dataX, literal);
	}

private:
	PTX::Register<PTX::PredicateType> *m_runningPredicate = nullptr;
	PTX::Register<D> *m_targetRegister = nullptr;

	PTX::Register<PTX::UInt32Type> *m_writeOffset = nullptr; // Join output

	const HorseIR::Operand *m_dataX = nullptr;

	FindOperation m_findOp;
	std::vector<ComparisonOperation> m_comparisonOps;
};

}
}
