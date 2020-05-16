#pragma once

#include "Codegen/Generators/Expressions/Builtins/BuiltinGenerator.h"

#include "Codegen/Builder.h"
#include "Codegen/Generators/Expressions/MoveGenerator.h"
#include "Codegen/Generators/Expressions/OperandGenerator.h"
#include "Codegen/Generators/Indexing/DataSizeGenerator.h"
#include "Codegen/Generators/Indexing/ThreadIndexGenerator.h"

#include "HorseIR/Tree/Tree.h"

#include "PTX/PTX.h"

namespace Codegen {

template<PTX::Bits B, class T>
class AppendGenerator : public BuiltinGenerator<B, T>
{
public:
	using BuiltinGenerator<B, T>::BuiltinGenerator;

	std::string Name() const override { return "AppendGenerator"; }

	const PTX::Register<PTX::PredicateType> *GenerateCompressionPredicate(const std::vector<HorseIR::Operand *>& arguments) override
	{
		// Requires synchronization-in, never compressed

		return nullptr;
	}

	const PTX::Register<T> *Generate(const HorseIR::LValue *target, const std::vector<HorseIR::Operand *>& arguments) override
	{
		auto resources = this->m_builder.GetLocalResources();

		// Load the first vector from the first X threads, and the second vector from the last Y threads (data indexes Y-X)

		auto targetRegister = this->GenerateTargetRegister(target, arguments);

		OperandGenerator<B, T> opGen(this->m_builder);
		auto data0 = opGen.GenerateOperand(arguments.at(0), OperandGenerator<B, T>::LoadKind::Vector);

		MoveGenerator<T> moveGenerator(this->m_builder);
		moveGenerator.Generate(targetRegister, data0);

		DataSizeGenerator<B> sizeGenerator(this->m_builder);
		auto size0 = sizeGenerator.GenerateSize(arguments.at(0));

		ThreadIndexGenerator<B> indexGenerator(this->m_builder);
		auto index = indexGenerator.GenerateGlobalIndex();

		auto offsetLabel = this->m_builder.CreateLabel("OFFSET");
		auto offsetPredicate = resources->template AllocateTemporary<PTX::PredicateType>();

		this->m_builder.AddStatement(new PTX::SetPredicateInstruction<PTX::UInt32Type>(
			offsetPredicate, index, size0, PTX::UInt32Type::ComparisonOperator::Less
		));
		this->m_builder.AddStatement(new PTX::BranchInstruction(offsetLabel, offsetPredicate));

		auto indexOffset = resources->template AllocateTemporary<PTX::UInt32Type>();
		this->m_builder.AddStatement(new PTX::SubtractInstruction<PTX::UInt32Type>(indexOffset, index, size0));

		auto data1 = opGen.GenerateOperand(arguments.at(1), indexOffset, this->m_builder.UniqueIdentifier("append"));
		moveGenerator.Generate(targetRegister, data1);

		this->m_builder.AddStatement(offsetLabel);

		return targetRegister;
	}
};

}
