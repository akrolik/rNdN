#pragma once

#include "Codegen/Generators/Expressions/Builtins/BuiltinGenerator.h"

#include "Analysis/Shape/Shape.h"
#include "Analysis/Shape/ShapeUtils.h"

#include "Codegen/Builder.h"
#include "Codegen/Generators/IndexGenerator.h"
#include "Codegen/Generators/Expressions/ConversionGenerator.h"
#include "Codegen/Generators/Expressions/OperandCompressionGenerator.h"
#include "Codegen/Generators/Expressions/OperandGenerator.h"

#include "HorseIR/Tree/Tree.h"

#include "PTX/PTX.h"

namespace Codegen {

template<PTX::Bits B, class T, typename Enable = void>
class WhereGenerator : public BuiltinGenerator<B, T>
{
public:
	using BuiltinGenerator<B, T>::BuiltinGenerator;
};

template<PTX::Bits B>
class WhereGenerator<B, PTX::Int64Type> : public BuiltinGenerator<B, PTX::Int64Type>
{
public:
	using BuiltinGenerator<B, PTX::Int64Type>::BuiltinGenerator;

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
		auto predicate = opGen.GenerateRegister(arguments.at(0), OperandGenerator<B, PTX::PredicateType>::LoadKind::Vector);

		// We cannot produce the indices of compressed data as this would prevent knowing the current index for each true data item.

		if (resources->template IsCompressedRegister<PTX::PredicateType>(predicate))
		{
			BuiltinGenerator<B, PTX::Int64Type>::Unimplemented("where operation on compressed data");
		}

		// Generate the index for the data item and convert to the right type

		IndexGenerator indexGenerator(this->m_builder);
		auto index = indexGenerator.GenerateDataIndex();

		GeometryGenerator geometryGenerator(this->m_builder);
		auto size = geometryGenerator.GenerateDataSize();

		// Copy the index to the data register

		//TODO: Handle double compression using a prefix sum to compute the index

		auto dataRegister = this->GenerateTargetRegister(target, arguments);
		ConversionGenerator::ConvertSource<PTX::Int64Type, PTX::UInt32Type>(this->m_builder, dataRegister, index);

		// Ensure that the predicate is false for out-of-bounds indexes

		auto boundedPredicate = resources->template AllocateTemporary<PTX::PredicateType>();
		this->m_builder.AddStatement(new PTX::SetPredicateInstruction<PTX::UInt32Type>(
			boundedPredicate, nullptr, index, size, PTX::UInt32Type::ComparisonOperator::Less, predicate, PTX::PredicateModifier::BoolOperator::And
		));

		resources->SetCompressedRegister(dataRegister, boundedPredicate);

		return dataRegister;
	}
};

}
