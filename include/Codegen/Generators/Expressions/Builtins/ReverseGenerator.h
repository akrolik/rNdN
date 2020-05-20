#pragma once

#include "Codegen/Generators/Expressions/Builtins/BuiltinGenerator.h"

#include "Codegen/Builder.h"
#include "Codegen/Generators/Expressions/MoveGenerator.h"
#include "Codegen/Generators/Expressions/OperandGenerator.h"
#include "Codegen/Generators/Indexing/DataIndexGenerator.h"
#include "Codegen/Generators/Indexing/DataSizeGenerator.h"

#include "HorseIR/Tree/Tree.h"

#include "PTX/PTX.h"

namespace Codegen {

template<PTX::Bits B, class T>
class ReverseGenerator : public BuiltinGenerator<B, T>
{
public:
	using BuiltinGenerator<B, T>::BuiltinGenerator;

	std::string Name() const override { return "ReverseGenerator"; }

	const PTX::Register<PTX::PredicateType> *GenerateCompressionPredicate(const std::vector<HorseIR::Operand *>& arguments) override
	{
		return OperandCompressionGenerator::UnaryCompressionRegister(this->m_builder, arguments);
	}

	const PTX::Register<T> *Generate(const HorseIR::LValue *target, const std::vector<HorseIR::Operand *>& arguments) override
	{
		OperandGenerator<B, T> opGen(this->m_builder);
		auto value = opGen.GenerateOperand(arguments.at(0), OperandGenerator<B, T>::LoadKind::Vector);

		// Generate copies of the data and index for writing

		const PTX::Register<PTX::UInt32Type> *index = nullptr;

		OperandCompressionGenerator compressionGenerator(this->m_builder);
		if (auto compressionPredicate = compressionGenerator.GetCompressionRegister(arguments.at(0)))
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

		DataSizeGenerator<B> sizeGenerator(this->m_builder);
		auto size = sizeGenerator.GenerateSize(arguments.at(0));

		auto resources = this->m_builder.GetLocalResources();
		auto writeIndex1 = resources->template AllocateTemporary<PTX::UInt32Type>();
		auto writeIndex2 = resources->template AllocateTemporary<PTX::UInt32Type>();

		this->m_builder.AddStatement(new PTX::SubtractInstruction<PTX::UInt32Type>(writeIndex1, size, index));
		this->m_builder.AddStatement(new PTX::SubtractInstruction<PTX::UInt32Type>(writeIndex2, writeIndex1, new PTX::UInt32Value(1)));

		// Move data

		auto targetRegister = this->GenerateTargetRegister(target, arguments);
		MoveGenerator<T> moveGenerator(this->m_builder);
		moveGenerator.Generate(targetRegister, value);

		// Set indexing used for writing output data

		resources->SetIndexedRegister(targetRegister, writeIndex2);

		return targetRegister;
	}
};

}