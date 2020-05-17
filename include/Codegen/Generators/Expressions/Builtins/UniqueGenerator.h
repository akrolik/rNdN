#pragma once

#include "Codegen/Generators/Generator.h"

#include "Codegen/Builder.h"
#include "Codegen/Generators/Data/TargetGenerator.h"
#include "Codegen/Generators/Expressions/OperandGenerator.h"
#include "Codegen/Generators/Expressions/Builtins/InternalChangeGenerator.h"
#include "Codegen/Generators/Indexing/DataIndexGenerator.h"

#include "HorseIR/Tree/Tree.h"

#include "PTX/PTX.h"

namespace Codegen {

template<PTX::Bits B>
class UniqueGenerator : public Generator
{
public:
	using Generator::Generator;

	std::string Name() const override { return "UniqueGenerator"; }

	void Generate(const std::vector<HorseIR::LValue *>& targets, const std::vector<HorseIR::Operand *>& arguments)
	{
		auto resources = this->m_builder.GetLocalResources();

		// Convenience split of input arguments

		const auto indexArgument = arguments.at(0);
		const auto dataArgument = arguments.at(1);

		// Check for each column if there has been a change

		InternalChangeGenerator<B> changeGenerator(this->m_builder);
		auto changePredicate = changeGenerator.Generate(dataArgument);

		// Get the return targets!

		TargetGenerator<B, PTX::Int64Type> targetGenerator(this->m_builder);
		auto keys = targetGenerator.Generate(targets.at(0), nullptr);

		// Set the key as the dataIndex value

		DataIndexGenerator<B> indexGenerator(this->m_builder);
		auto index = indexGenerator.GenerateDataIndex();

		OperandGenerator<B, PTX::Int64Type> operandGenerator(this->m_builder);
		auto dataIndex = operandGenerator.GenerateOperand(indexArgument, index, this->m_builder.UniqueIdentifier("index"));

		this->m_builder.AddStatement(new PTX::MoveInstruction<PTX::Int64Type>(keys, dataIndex));
		resources->SetCompressedRegister(keys, changePredicate);
	}
};

}