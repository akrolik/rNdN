#pragma once

#include "Codegen/Generators/Expressions/Builtins/BuiltinGenerator.h"

#include "Codegen/Builder.h"
#include "Codegen/Generators/Expressions/MoveGenerator.h"
#include "Codegen/Generators/Expressions/OperandGenerator.h"
#include "Codegen/Generators/Expressions/OperandCompressionGenerator.h"
#include "Codegen/Generators/Indexing/DataIndexGenerator.h"
#include "Codegen/Generators/Indexing/PrefixSumGenerator.h"

#include "HorseIR/Tree/Tree.h"

#include "PTX/Tree/Tree.h"

namespace Codegen {

enum class LimitOperation {
	Drop,
	Take
};

static std::string LimitOperationString(LimitOperation limitOp)
{
	switch (limitOp)
	{
		case LimitOperation::Drop:
			return "drop";
		case LimitOperation::Take:
			return "take";
	}
	return "<unknown>";
}

template<PTX::Bits B, class T>
class LimitGenerator : public BuiltinGenerator<B, T>
{
public:
	LimitGenerator(Builder& builder, LimitOperation limitOp) : BuiltinGenerator<B, T>(builder), m_limitOp(limitOp) {}

	std::string Name() const override { return "LimitGenerator"; }

	// The output of a limit function handles the predicate itself. We therefore do not implement GenerateCompressionPredicate in this subclass

	const PTX::Register<T> *Generate(const HorseIR::LValue *target, const std::vector<HorseIR::Operand *>& arguments) override
	{
		auto resources = this->m_builder.GetLocalResources();

		// Range in argument 0, value in argument 1
		// Move data verbatim to the destination register

		OperandGenerator<B, T> operandGenerator(this->m_builder);
		auto value = operandGenerator.GenerateOperand(arguments.at(1), OperandGenerator<B, T>::LoadKind::Vector);
		auto targetRegister = this->GenerateTargetRegister(target, arguments);

		DataSizeGenerator<B> sizeGenerator(this->m_builder);
		auto size = sizeGenerator.GenerateSize(arguments.at(1));

		MoveGenerator<T> moveGenerator(this->m_builder);
		moveGenerator.Generate(targetRegister, value);

		// Get index and compression used for writing output data

		OperandGenerator<B, PTX::Int32Type> operandLimitGenerator(this->m_builder);
		auto limit = operandLimitGenerator.GenerateOperand(arguments.at(0), OperandGenerator<B, PTX::Int32Type>::LoadKind::Vector);

		// Take function
		//   - if n > 0: n leading items
		//   - if n < 0: n trailing items
		//   - Special case: if n > length, initialized to zero
		//
		// Drop function:
		//   - if n > 0: drops n leading items
		//   - if n < 0: drops n trailing items
		//   - Special case: if n >= length, empty vector

		auto lowerBound = resources->template AllocateTemporary<PTX::UInt32Type>();
		auto upperBound = resources->template AllocateTemporary<PTX::UInt32Type>();

		auto switchPredicate = resources->template AllocateTemporary<PTX::PredicateType>();
		this->m_builder.AddStatement(new PTX::SetPredicateInstruction<PTX::Int32Type>(switchPredicate, limit, new PTX::Int32Value(0), PTX::Int32Type::ComparisonOperator::Less));

		auto absLimit = resources->template AllocateTemporary<PTX::UInt32Type>();
		this->m_builder.AddStatement(new PTX::AbsoluteInstruction<PTX::Int32Type>(new PTX::Signed32RegisterAdapter(absLimit), limit));

		auto elseLabel = this->m_builder.CreateLabel("ELSE");
		auto endLabel = this->m_builder.CreateLabel("END");

		this->m_builder.AddStatement(new PTX::BranchInstruction(elseLabel, switchPredicate));

		GeneratePositiveBounds(lowerBound, upperBound, size, absLimit);
		this->m_builder.AddStatement(new PTX::BranchInstruction(endLabel));

		this->m_builder.AddStatement(elseLabel);
		GenerateNegativeBounds(lowerBound, upperBound, size, absLimit);

		this->m_builder.AddStatement(endLabel);

		// Compute the data indexes that will be used for bounds checking

		OperandCompressionGenerator compressionGenerator(this->m_builder);
		auto compressionPredicate = compressionGenerator.GetCompressionRegister(arguments.at(1));

		const PTX::Register<PTX::UInt32Type> *index = nullptr;
		if (compressionPredicate != nullptr)
		{
			// Compression requires a prefix sum to compute the index

			PrefixSumGenerator<B, PTX::UInt32Type> prefixSumGenerator(this->m_builder);
			index = prefixSumGenerator.template Generate<PTX::PredicateType>(compressionPredicate, PrefixSumMode::Exclusive);
		}
		else
		{
			DataIndexGenerator<B> indexGenerator(this->m_builder);
			index = indexGenerator.GenerateDataIndex();
		}

		// Compute the bounds checks

		auto upperPredicate = resources->template AllocateTemporary<PTX::PredicateType>();
		auto lowerPredicate = resources->template AllocateTemporary<PTX::PredicateType>();
		auto limitPredicate = resources->template AllocateTemporary<PTX::PredicateType>();

		this->m_builder.AddStatement(new PTX::SetPredicateInstruction<PTX::UInt32Type>(lowerPredicate, index, lowerBound, PTX::UInt32Type::ComparisonOperator::GreaterEqual));
		this->m_builder.AddStatement(new PTX::SetPredicateInstruction<PTX::UInt32Type>(upperPredicate, index, upperBound, PTX::UInt32Type::ComparisonOperator::Less));
		this->m_builder.AddStatement(new PTX::AndInstruction<PTX::PredicateType>(limitPredicate, lowerPredicate, upperPredicate));

		// Compute write index (offset by the lower bound)

		auto writeIndex = resources->template AllocateTemporary<PTX::UInt32Type>();
		this->m_builder.AddStatement(new PTX::SubtractInstruction<PTX::UInt32Type>(writeIndex, index, lowerBound));
		resources->SetIndexedRegister(targetRegister, writeIndex);
		
		// Set compression to only include live indexes

		if (compressionPredicate != nullptr)
		{
			auto mergedPredicate = resources->template AllocateTemporary<PTX::PredicateType>();
			this->m_builder.AddStatement(new PTX::AndInstruction<PTX::PredicateType>(mergedPredicate, limitPredicate, compressionPredicate));
			resources->SetCompressedRegister(targetRegister, mergedPredicate);
		}
		else
		{
			resources->SetCompressedRegister(targetRegister, limitPredicate);
		}

		return targetRegister;
	}

private:
	void GeneratePositiveBounds(
		const PTX::Register<PTX::UInt32Type> *lowerBound, const PTX::Register<PTX::UInt32Type> *upperBound,
		const PTX::TypedOperand<PTX::UInt32Type> *size, const PTX::TypedOperand<PTX::UInt32Type> *limit
	)
	{
		switch (m_limitOp)
		{
			case LimitOperation::Take:
			{
				this->m_builder.AddStatement(new PTX::MoveInstruction<PTX::UInt32Type>(lowerBound, new PTX::UInt32Value(0)));
				this->m_builder.AddStatement(new PTX::MoveInstruction<PTX::UInt32Type>(upperBound, limit));
				break;
			}
			case LimitOperation::Drop:
			{
				this->m_builder.AddStatement(new PTX::MoveInstruction<PTX::UInt32Type>(lowerBound, limit));
				this->m_builder.AddStatement(new PTX::MoveInstruction<PTX::UInt32Type>(upperBound, size));
				break;
			}
			default:
			{
				BuiltinGenerator<B, T>::Unimplemented("limit operation " + LimitOperationString(m_limitOp));
			}
		}
	}

	void GenerateNegativeBounds(
		const PTX::Register<PTX::UInt32Type> *lowerBound, const PTX::Register<PTX::UInt32Type> *upperBound,
		const PTX::TypedOperand<PTX::UInt32Type> *size, const PTX::TypedOperand<PTX::UInt32Type> *limit
	)
	{
		auto resources = this->m_builder.GetLocalResources();
		auto sizeMLimit = resources->template AllocateTemporary<PTX::UInt32Type>();
		this->m_builder.AddStatement(new PTX::SubtractInstruction<PTX::UInt32Type>(sizeMLimit, size, limit));

		switch (m_limitOp)
		{
			case LimitOperation::Take:
			{
				this->m_builder.AddStatement(new PTX::MoveInstruction<PTX::UInt32Type>(lowerBound, sizeMLimit));
				this->m_builder.AddStatement(new PTX::MoveInstruction<PTX::UInt32Type>(upperBound, size));
				break;
			}
			case LimitOperation::Drop:
			{
				this->m_builder.AddStatement(new PTX::MoveInstruction<PTX::UInt32Type>(lowerBound, new PTX::UInt32Value(0)));
				this->m_builder.AddStatement(new PTX::MoveInstruction<PTX::UInt32Type>(upperBound, sizeMLimit));
				break;
			}
			default:
			{
				BuiltinGenerator<B, T>::Unimplemented("limit operation " + LimitOperationString(m_limitOp));
			}
		}
	}

	LimitOperation m_limitOp;
};

}
