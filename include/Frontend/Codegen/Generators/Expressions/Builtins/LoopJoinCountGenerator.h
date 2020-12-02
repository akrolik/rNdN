#pragma once

#include "Frontend/Codegen/Generators/Generator.h"

#include "Frontend/Codegen/Builder.h"
#include "Frontend/Codegen/Generators/Data/TargetGenerator.h"
#include "Frontend/Codegen/Generators/Expressions/Builtins/ComparisonGenerator.h"
#include "Frontend/Codegen/Generators/Expressions/Builtins/InternalFindGenerator.h"
#include "Frontend/Codegen/Generators/Indexing/DataIndexGenerator.h"
#include "Frontend/Codegen/Generators/Indexing/DataSizeGenerator.h"
#include "Frontend/Codegen/Generators/Indexing/PrefixSumGenerator.h"
#include "Frontend/Codegen/Generators/Indexing/ThreadIndexGenerator.h"

#include "HorseIR/Tree/Tree.h"
#include "HorseIR/Utils/PrettyPrinter.h"
#include "HorseIR/Utils/TypeUtils.h"

#include "PTX/Tree/Tree.h"

namespace Frontend {
namespace Codegen {

template<PTX::Bits B>
class LoopJoinCountGenerator : public Generator
{
public:
	using Generator::Generator;

	std::string Name() const override { return "LoopJoinCountGenerator"; }

	void Generate(const std::vector<HorseIR::LValue *>& targets, const std::vector<HorseIR::Operand *>& arguments)
	{
		std::vector<HorseIR::Operand *> functionArguments(std::begin(arguments), std::end(arguments) - 2);
		std::vector<HorseIR::Operand *> dataArguments(std::begin(arguments) + functionArguments.size(), std::end(arguments));

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

		InternalFindGenerator<B, PTX::Int64Type> findGenerator(this->m_builder, FindOperation::Count, joinOperations);
		auto offsetsRegister = findGenerator.Generate(targets.at(0), {dataArguments.at(1), dataArguments.at(0)});

		// Compute prefix sum, getting the offset for each thread

		PrefixSumGenerator<B, PTX::Int64Type> prefixSumGenerator(this->m_builder);
		auto prefixSum = prefixSumGenerator.Generate(offsetsRegister, PrefixSumMode::Exclusive);

		// Compute the count of data items

		TargetGenerator<B, PTX::Int64Type> targetGenerator(this->m_builder);
		auto countRegister = targetGenerator.Generate(targets.at(1), nullptr);

		this->m_builder.AddStatement(new PTX::AddInstruction<PTX::Int64Type>(countRegister, offsetsRegister, prefixSum));

		// Only store for the last thread

		auto resources = this->m_builder.GetLocalResources();
		auto countPredicate = resources->template AllocateTemporary<PTX::PredicateType>();

		ThreadIndexGenerator<B> indexGenerator(this->m_builder);
		auto index = indexGenerator.GenerateGlobalIndex();

		DataSizeGenerator<B> sizeGenerator(this->m_builder);
		auto size = sizeGenerator.GenerateSize(dataArguments.at(1));
		auto sizeLess1 = resources->template AllocateTemporary<PTX::UInt32Type>();

		this->m_builder.AddStatement(new PTX::SubtractInstruction<PTX::UInt32Type>(sizeLess1, size, new PTX::UInt32Value(1)));
		this->m_builder.AddStatement(new PTX::SetPredicateInstruction<PTX::UInt32Type>(countPredicate, index, sizeLess1, PTX::UInt32Type::ComparisonOperator::Equal));

		resources->SetCompressedRegister<PTX::Int64Type>(countRegister, countPredicate);
		resources->SetIndexedRegister<PTX::Int64Type>(countRegister, new PTX::UInt32Value(0));

		// Generate output

		this->m_builder.AddStatement(new PTX::MoveInstruction<PTX::Int64Type>(offsetsRegister, prefixSum));
	}
};

}
}
