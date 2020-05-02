#pragma once

#include "HorseIR/Traversal/ConstVisitor.h"
#include "Codegen/Generators/Expressions/Builtins/BuiltinGenerator.h"

#include "Codegen/Builder.h"
#include "Codegen/Generators/Expressions/OperandCompressionGenerator.h"
#include "Codegen/Generators/Expressions/OperandGenerator.h"
#include "Codegen/Generators/Expressions/Builtins/ComparisonGenerator.h"
#include "Codegen/Generators/Indexing/AddressGenerator.h"
#include "Codegen/Generators/Indexing/DataSizeGenerator.h"
#include "Codegen/Generators/Indexing/ThreadIndexGenerator.h"
#include "Codegen/Generators/Synchronization/BarrierGenerator.h"

#include "HorseIR/Tree/Tree.h"

#include "PTX/PTX.h"

namespace Codegen {

enum class FindOperation {
	Member,
	Index,
	Count
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
		m_targetRegister = this->GenerateTargetRegister(target, arguments);

		// Initialize target register (@member->false, @index_of->0, @join_count->0)

		if constexpr(std::is_same<D, PTX::PredicateType>::value)
		{
			if (m_findOp == FindOperation::Member)
			{
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
					m_predicateRegister = resources->template AllocateTemporary<PTX::PredicateType>();

					this->m_builder.AddStatement(new PTX::MoveInstruction<PTX::PredicateType>(m_predicateRegister, new PTX::BoolValue(false)));
					// Fallthrough
				}
				case FindOperation::Count:
				{
					this->m_builder.AddStatement(new PTX::MoveInstruction<PTX::Int64Type>(m_targetRegister, new PTX::Int64Value(0)));
				}
			}
		}

		// Operating range in argument 0, possibilities in argument 1

		m_data = arguments.at(0);
		arguments.at(1)->Accept(*this);

		return m_targetRegister;
	}

	void Visit(const HorseIR::Identifier *identifier) override
	{
		DispatchType(*this, identifier->GetType(), identifier);
	}

	template<class T>
	void GenerateList(const HorseIR::Identifier *identifier)
	{
		//TODO: List match
	}

	template<class T>
	void GenerateTuple(unsigned int index, const HorseIR::Identifier *identifier)
	{
		//TODO: Tuple match
	}

	template<class T>
	void GenerateVector(const HorseIR::Identifier *identifier)
	{
		// Generate a double loop, checking for matches in chunks (for coalescing)
		//
		// __shared__ T s[BLOCK_SIZE]
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
		//    <increment, i, BLOCK_SIZE>
		//
		//    br START_1
		//
		// END_1:

		if constexpr(std::is_same<T, PTX::PredicateType>::value || std::is_same<T, PTX::Int8Type>::value)
		{
			//TODO: Supported shared memory for smaller types
		}
		else
		{
			auto resources = this->m_builder.GetLocalResources();
			auto kernelResources = this->m_builder.GetKernelResources();
			auto globalResources = this->m_builder.GetGlobalResources();

			// Maximize the block size

			auto& targetOptions = this->m_builder.GetTargetOptions();
			auto& kernelOptions = this->m_builder.GetKernelOptions();

			constexpr auto BLOCK_SIZE = 1024u;
			kernelOptions.SetBlockSize(BLOCK_SIZE);

			auto s_cache = new PTX::ArrayVariableAdapter<T, BLOCK_SIZE, PTX::SharedSpace>(
				kernelResources->template AllocateSharedVariable<PTX::ArrayType<T, BLOCK_SIZE>>(this->m_builder.UniqueIdentifier("cache"))
			);

			AddressGenerator<B, T> addressGenerator(this->m_builder);
			BarrierGenerator<B> barrierGenerator(this->m_builder);

			ThreadIndexGenerator<B> threadGenerator(this->m_builder);
			DataSizeGenerator<B> sizeGenerator(this->m_builder);

			// Get the data

			OperandGenerator<B, T> opGen(this->m_builder);
			auto data = opGen.GenerateOperand(m_data, OperandGenerator<B, T>::LoadKind::Vector);

			// Initialize the outer loop

			auto size = sizeGenerator.GenerateSize(identifier);
			auto index = threadGenerator.GenerateLocalIndex();

			auto bound = resources->template AllocateTemporary<PTX::UInt32Type>();
			this->m_builder.AddStatement(new PTX::AddInstruction<PTX::UInt32Type>(bound, index, size));

			// Check if the final iteration is complete

			auto remainder = resources->template AllocateTemporary<PTX::UInt32Type>();
			this->m_builder.AddStatement(new PTX::RemainderInstruction<PTX::UInt32Type>(remainder, size, new PTX::UInt32Value(BLOCK_SIZE)));

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

			// Begin the loop body
			//   - Load data from global into shared
			//   - Synchronize

			OperandGenerator<B, T> operandGenerator(this->m_builder);
			auto value = operandGenerator.GenerateRegister(identifier, index, this->m_builder.UniqueIdentifier("member"));

			auto localIndex = threadGenerator.GenerateLocalIndex();
			auto s_cacheAddress = addressGenerator.GenerateAddress(s_cache, localIndex);

			this->m_builder.AddStatement(new PTX::StoreInstruction<B, T, PTX::SharedSpace>(s_cacheAddress, value));

			barrierGenerator.Generate();

			// Setup the inner loop and bound
			//  - Chunks 0...(N-1): BLOCK_SIZE
			//  - Chunk N: remainder

			// Increment by the number of threads. Do it here since it will be reused

			this->m_builder.AddStatement(new PTX::AddInstruction<PTX::UInt32Type>(index, index, new PTX::UInt32Value(BLOCK_SIZE)));

			// Chose the correct bound for the inner loop, sizePredicate indicates a complete iteration

			auto sizePredicate_2 = resources->template AllocateTemporary<PTX::PredicateType>();
			auto sizePredicate_3 = resources->template AllocateTemporary<PTX::PredicateType>();

			this->m_builder.AddStatement(new PTX::SetPredicateInstruction<PTX::UInt32Type>(sizePredicate_2, index, bound, PTX::UInt32Type::ComparisonOperator::GreaterEqual));
			this->m_builder.AddStatement(new PTX::AndInstruction<PTX::PredicateType>(sizePredicate_3, sizePredicate_1, sizePredicate_2));

			auto ifElseLabel = this->m_builder.CreateLabel("ELSE"); 
			auto ifEndLabel = this->m_builder.CreateLabel("END"); 

			this->m_builder.AddStatement(new PTX::BranchInstruction(ifElseLabel, sizePredicate_3));
			this->m_builder.AddStatement(new PTX::BlankStatement());

			GenerateInnerLoop(data, s_cache, new PTX::UInt32Value(BLOCK_SIZE), BLOCK_SIZE, 16);

			this->m_builder.AddStatement(new PTX::BranchInstruction(ifEndLabel));
			this->m_builder.AddStatement(new PTX::BlankStatement());
			this->m_builder.AddStatement(ifElseLabel);

			// Compute bound if this is the last iteration

			GenerateInnerLoop(data, s_cache, remainder, BLOCK_SIZE, 1);
			
			this->m_builder.AddStatement(ifEndLabel);

			// Barrier before the next chunk, otherwise some warps might overwrite the previous values too soon

			barrierGenerator.Generate();

			// Complete the loop structure, check the next index chunk

			this->m_builder.AddStatement(new PTX::BranchInstruction(startLabel));
			this->m_builder.AddStatement(new PTX::BlankStatement());
			this->m_builder.AddStatement(endLabel);
		}
	}

	template<class T>
	void GenerateInnerLoop(const PTX::TypedOperand<T> *data, const PTX::SharedVariable<T> *s_cache, const PTX::TypedOperand<PTX::UInt32Type> *bound, unsigned int BLOCK_SIZE, unsigned int factor)
	{
		auto resources = this->m_builder.GetLocalResources();

		// Loop body
		//
		//    j = 0
		//    START_2: (unroll)
		//       setp %p2, j, BLOCK_SIZE
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
		
		auto base = resources->template AllocateTemporary<PTX::UIntType<B>>();
		auto basePointer = new PTX::PointerRegisterAdapter<B, T, PTX::SharedSpace>(base);
		auto baseAddress = new PTX::MemoryAddress<B, T, PTX::SharedSpace>(s_cache);

		this->m_builder.AddStatement(new PTX::MoveAddressInstruction<B, T, PTX::SharedSpace>(basePointer, baseAddress));

		auto startLabel = this->m_builder.CreateLabel("START");
		auto endLabel = this->m_builder.CreateLabel("END");
		auto predicate = resources->template AllocateTemporary<PTX::PredicateType>();

		this->m_builder.AddStatement(startLabel);
		this->m_builder.AddStatement(new PTX::SetPredicateInstruction<PTX::UInt32Type>(predicate, index, bound, PTX::UInt32Type::ComparisonOperator::GreaterEqual));
		this->m_builder.AddStatement(new PTX::BranchInstruction(endLabel, predicate));
		this->m_builder.AddStatement(new PTX::BlankStatement());

		for (auto i = 0; i < factor; ++i)
		{
			// Inner loop body, match the value against the thread data

			auto value = resources->template AllocateTemporary<T>();
			auto s_cacheAddress = new PTX::RegisterAddress<B, T, PTX::SharedSpace>(basePointer, i);

			this->m_builder.AddStatement(new PTX::LoadInstruction<B, T, PTX::SharedSpace>(value, s_cacheAddress));

			GenerateMatch(data, value);
		}

		// Increment by the iterations completed

		this->m_builder.AddStatement(new PTX::AddInstruction<PTX::UInt32Type>(index, index, new PTX::UInt32Value(factor)));
		this->m_builder.AddStatement(new PTX::AddInstruction<PTX::UIntType<B>>(base, base, new PTX::UIntValue<B>(factor * PTX::BitSize<T::TypeBits>::NumBytes)));

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
	template<class T>
	void GenerateMatch(const PTX::TypedOperand<T> *data, const PTX::TypedOperand<T> *value)
	{
		auto resources = this->m_builder.GetLocalResources();
		auto predicate = resources->template AllocateTemporary<PTX::PredicateType>();

		ComparisonGenerator<B, PTX::PredicateType> comparisonGenerator(this->m_builder, m_comparisonOps.at(0));
		comparisonGenerator.Generate(predicate, data, value);

		if constexpr(std::is_same<D, PTX::PredicateType>::value)
		{
			if (m_findOp == FindOperation::Member)
			{
				this->m_builder.AddStatement(new PTX::OrInstruction<PTX::PredicateType>(m_targetRegister, m_targetRegister, predicate));
			}
		}
		else if constexpr(std::is_same<D, PTX::Int64Type>::value)
		{
			switch (m_findOp)
			{
				case FindOperation::Index:
				{
					this->m_builder.AddStatement(new PTX::OrInstruction<PTX::PredicateType>(m_predicateRegister, m_predicateRegister, predicate));

					auto addInstruction = new PTX::AddInstruction<PTX::Int64Type>(m_targetRegister, m_targetRegister, new PTX::Int64Value(1));
					addInstruction->SetPredicate(m_predicateRegister, true);
					this->m_builder.AddStatement(addInstruction);

					break;
				}
				case FindOperation::Count:
				{
					auto addInstruction = new PTX::AddInstruction<PTX::Int64Type>(m_targetRegister, m_targetRegister, new PTX::Int64Value(1));
					addInstruction->SetPredicate(predicate);
					this->m_builder.AddStatement(addInstruction);

					break;
				}
			}
		}
	}

	const PTX::Register<PTX::PredicateType> *m_predicateRegister = nullptr;
	const PTX::Register<D> *m_targetRegister = nullptr;

	const HorseIR::Operand *m_data = nullptr;

	FindOperation m_findOp;
	std::vector<ComparisonOperation> m_comparisonOps;
};

}
