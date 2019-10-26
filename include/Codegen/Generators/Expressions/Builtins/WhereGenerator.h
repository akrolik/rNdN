#pragma once

#include "Codegen/Generators/Expressions/Builtins/BuiltinGenerator.h"

#include "Codegen/Builder.h"
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
		auto dataRegister = this->GenerateTargetRegister(target, arguments);

		auto& inputOptions = this->m_builder.GetInputOptions();
		if (Analysis::ShapeUtils::IsShape<Analysis::VectorShape>(inputOptions.ThreadGeometry))
		{
			auto index = indexGenerator.GenerateGlobalIndex();
			ConversionGenerator::ConvertSource<PTX::Int64Type, PTX::UInt32Type>(this->m_builder, dataRegister, index);
		}
		else if (Analysis::ShapeUtils::IsShape<Analysis::ListShape>(inputOptions.ThreadGeometry))
		{
			auto index = indexGenerator.GenerateCellDataIndex();
			ConversionGenerator::ConvertSource<PTX::Int64Type, PTX::UInt32Type>(this->m_builder, dataRegister, index);
		}
		else
		{
			BuiltinGenerator<B, PTX::Int64Type>::Unimplemented("where operation for thread geometry " + Analysis::ShapeUtils::ShapeString(inputOptions.ThreadGeometry));
		}

		// Copy the compression mask in case it is reasigned

		auto predicateTemp = resources->template AllocateTemporary<PTX::PredicateType>();
		this->m_builder.AddStatement(new PTX::MoveInstruction<PTX::PredicateType>(predicateTemp, predicate));

		resources->SetCompressedRegister(dataRegister, predicateTemp);

		return dataRegister;
	}
};

}
