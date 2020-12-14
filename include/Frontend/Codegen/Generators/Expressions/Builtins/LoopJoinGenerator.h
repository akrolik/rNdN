#pragma once

#include "Frontend/Codegen/Generators/Generator.h"

#include "Frontend/Codegen/Builder.h"
#include "Frontend/Codegen/Generators/Data/TargetCellGenerator.h"
#include "Frontend/Codegen/Generators/Expressions/Builtins/ComparisonGenerator.h"

#include "HorseIR/Tree/Tree.h"

#include "PTX/Tree/Tree.h"

namespace Frontend {
namespace Codegen {

template<PTX::Bits B>
class LoopJoinGenerator : public Generator
{
public:
	using Generator::Generator;

	std::string Name() const override { return "LoopJoinGenerator"; }

	void Generate(const HorseIR::LValue *target, const std::vector<const HorseIR::Operand *>& arguments)
	{
		std::vector<const HorseIR::Operand *> functionArguments(std::begin(arguments), std::end(arguments) - 4);
		std::vector<const HorseIR::Operand *> dataArguments(std::begin(arguments) + functionArguments.size(), std::end(arguments));

		std::vector<ComparisonOperation> joinOperations;
		for (auto functionArgument : functionArguments)
		{
			if (auto functionType = HorseIR::TypeUtils::GetType<HorseIR::FunctionType>(functionArgument->GetType()))
			{
				auto joinOperation = GetJoinComparisonOperation(functionType->GetFunctionDeclaration(), true);
				joinOperations.push_back(joinOperation);
			}
			else
			{
				Generator::Error("non-function join argument '" + HorseIR::PrettyPrinter::PrettyString(functionArgument, true) + "'");
			}
		}

		// Count the number of join results per left-hand data item

		InternalFindGenerator<B, PTX::Int64Type> findGenerator(this->m_builder, FindOperation::Indexes, joinOperations);
		findGenerator.Generate(target, {dataArguments.at(1), dataArguments.at(0), dataArguments.at(2)});
	}
};

}
}
