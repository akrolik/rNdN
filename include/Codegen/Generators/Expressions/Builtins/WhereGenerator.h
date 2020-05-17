#pragma once

#include "Codegen/Generators/Expressions/Builtins/BuiltinGenerator.h"

#include "Codegen/Builder.h"
#include "Codegen/Generators/Expressions/ConversionGenerator.h"
#include "Codegen/Generators/Expressions/OperandGenerator.h"
#include "Codegen/Generators/Indexing/DataIndexGenerator.h"
#include "Codegen/Generators/Indexing/PrefixSumGenerator.h"
#include "Codegen/Generators/Indexing/ThreadGeometryGenerator.h"

#include "HorseIR/Tree/Tree.h"

#include "PTX/PTX.h"

namespace Codegen {

template<PTX::Bits B, class T, typename Enable = void>
class WhereGenerator : public BuiltinGenerator<B, T>
{
public:
	using BuiltinGenerator<B, T>::BuiltinGenerator;
	std::string Name() const override { return "WhereGenerator"; }
};

template<PTX::Bits B>
class WhereGenerator<B, PTX::Int64Type> : public BuiltinGenerator<B, PTX::Int64Type>
{
public:
	using BuiltinGenerator<B, PTX::Int64Type>::BuiltinGenerator;

	std::string Name() const override { return "WhereGenerator"; }

	// The output of a where function handles the predicate itself. We therefore do not implement GenerateCompressionPredicate in this subclass

	const PTX::Register<PTX::Int64Type> *Generate(const HorseIR::LValue *target, const std::vector<HorseIR::Operand *>& arguments) override
	{
		auto resources = this->m_builder.GetLocalResources();

		// Where produces a compressed list of indices, creating a (data, predicate) pair. It it thus similar
		// to compression with system provided values.
		// 
		// i.e. Given a where call
		//
		//         t1:i64 = @where(p);
		//
		// the pair (index, p) is created in the resource allocator. Future lookups for t1 will produce the index register.

		// Generate the predicate from the input data

		OperandGenerator<B, PTX::PredicateType> opGen(this->m_builder);
		auto inputPredicate = opGen.GenerateRegister(arguments.at(0), OperandGenerator<B, PTX::PredicateType>::LoadKind::Vector);

		// Generate the index for the data item and convert to the right type

		DataIndexGenerator<B> indexGenerator(this->m_builder);
		auto index = indexGenerator.GenerateDataIndex();

		auto dataRegister = this->GenerateTargetRegister(target, arguments);
		auto dataPredicate = inputPredicate; // Modified by the compression

		if (const auto compressed = resources->template GetCompressedRegister<PTX::PredicateType>(inputPredicate))
		{
			// Compression requires a prefix sum to compute the index

			PrefixSumGenerator<B, PTX::Int64Type> prefixSumGenerator(this->m_builder);
			auto offsetIndex = prefixSumGenerator.template Generate<PTX::PredicateType>(compressed, PrefixSumMode::Exclusive);

			this->m_builder.AddStatement(new PTX::MoveInstruction<PTX::Int64Type>(dataRegister, offsetIndex));

			dataPredicate = resources->template AllocateTemporary<PTX::PredicateType>();
			this->m_builder.AddStatement(new PTX::AndInstruction<PTX::PredicateType>(dataPredicate, inputPredicate, compressed));
		}
		else
		{
			ConversionGenerator::ConvertSource<PTX::Int64Type, PTX::UInt32Type>(this->m_builder, dataRegister, index);
		}

		// Ensure that the predicate is false for out-of-bounds indexes

		ThreadGeometryGenerator<B> geometryGenerator(this->m_builder);
		auto size = geometryGenerator.GenerateDataGeometry();

		auto boundsPredicate = resources->template AllocateTemporary<PTX::PredicateType>();
		auto boundedPredicate = resources->template AllocateTemporary<PTX::PredicateType>();

		this->m_builder.AddStatement(new PTX::SetPredicateInstruction<PTX::UInt32Type>(boundedPredicate, index, size, PTX::UInt32Type::ComparisonOperator::Less));
		this->m_builder.AddStatement(new PTX::AndInstruction<PTX::PredicateType>(boundedPredicate, boundsPredicate, dataPredicate));

		resources->SetCompressedRegister(dataRegister, boundedPredicate);

		return dataRegister;
	}
};

}
