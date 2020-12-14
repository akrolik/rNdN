#pragma once

#include "Frontend/Codegen/Generators/Generator.h"

#include "Frontend/Codegen/Builder.h"
#include "Frontend/Codegen/Generators/Data/TargetGenerator.h"
#include "Frontend/Codegen/Generators/Expressions/OperandGenerator.h"
#include "Frontend/Codegen/Generators/Expressions/Builtins/InternalChangeGenerator.h"
#include "Frontend/Codegen/Generators/Indexing/DataIndexGenerator.h"

#include "HorseIR/Tree/Tree.h"

#include "PTX/Tree/Tree.h"

namespace Frontend {
namespace Codegen {

template<PTX::Bits B>
class GroupGenerator : public Generator
{
public:
	using Generator::Generator;

	std::string Name() const override { return "GroupGenerator"; }

	void Generate(const std::vector<const HorseIR::LValue *>& targets, const std::vector<const HorseIR::Operand *>& arguments)
	{
		auto resources = this->m_builder.GetLocalResources();

		// Convenience split of input arguments

		const auto indexArgument = arguments.at(0);
		const auto dataArgument = arguments.at(1);

		// Initialize the current and previous values, and compute the change
		//   
		//   1. Check if geometry is within bounds
		//   2. Load the current value
		//   3. Load the previous value at index -1 (bounded below by index 0)

		// Check for each column if there has been a change

		InternalChangeGenerator<B> changeGenerator(this->m_builder);
		auto changePredicate = changeGenerator.Generate(dataArgument);

		// Get the return targets!

		TargetGenerator<B, PTX::Int64Type> targetGenerator(this->m_builder);
		auto keys = targetGenerator.Generate(targets.at(0), nullptr);
		auto values = targetGenerator.Generate(targets.at(1), nullptr);

		// Set the key as the dataIndex value

		DataIndexGenerator<B> indexGenerator(this->m_builder);
		auto index = indexGenerator.GenerateDataIndex();

		OperandGenerator<B, PTX::Int64Type> operandGenerator(this->m_builder);
		auto dataIndex = operandGenerator.GenerateOperand(indexArgument, index, this->m_builder.UniqueIdentifier("index"));

		this->m_builder.AddStatement(new PTX::MoveInstruction<PTX::Int64Type>(keys, dataIndex));
		resources->SetCompressedRegister(keys, changePredicate);

		// Set the value as the index into the dataIndex

		ConversionGenerator::ConvertSource(this->m_builder, values, index);
		resources->SetCompressedRegister(values, changePredicate);
	}
};

}
}
