#pragma once

#include "Codegen/Generators/Generator.h"

#include "Codegen/Builder.h"
#include "Codegen/Generators/Data/TargetGenerator.h"
#include "Codegen/Generators/Expressions/Builtins/InternalFindGenerator.h"
#include "Codegen/Generators/Indexing/DataIndexGenerator.h"
#include "Codegen/Generators/Indexing/DataSizeGenerator.h"
#include "Codegen/Generators/Indexing/PrefixSumGenerator.h"
#include "Codegen/Generators/Indexing/ThreadIndexGenerator.h"

#include "HorseIR/Tree/Tree.h"
#include "HorseIR/Utils/PrettyPrinter.h"
#include "HorseIR/Utils/TypeUtils.h"

#include "PTX/PTX.h"

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
				auto joinOperation = JoinOperation(functionType->GetFunctionDeclaration());
				joinOperations.push_back(joinOperation);
			}
			else
			{
				Generator::Error("non-function join argument '" + HorseIR::PrettyPrinter::PrettyString(functionArgument, true) + "'");
			}
		}

		// Count the number of join results per left-hand data item

		InternalFindGenerator<B, PTX::Int64Type> findGenerator(this->m_builder, FindOperation::Count, joinOperations);
		auto offsetsRegister = findGenerator.Generate(targets.at(0), dataArguments);

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
		auto size = sizeGenerator.GenerateSize(dataArguments.at(0));
		auto sizeLess1 = resources->template AllocateTemporary<PTX::UInt32Type>();

		this->m_builder.AddStatement(new PTX::SubtractInstruction<PTX::UInt32Type>(sizeLess1, size, new PTX::UInt32Value(1)));
		this->m_builder.AddStatement(new PTX::SetPredicateInstruction<PTX::UInt32Type>(countPredicate, index, sizeLess1, PTX::UInt32Type::ComparisonOperator::Equal));

		resources->SetCompressedRegister<PTX::Int64Type>(countRegister, countPredicate);
		resources->SetIndexedRegister<PTX::Int64Type>(countRegister, new PTX::UInt32Value(0));

		// Generate output

		this->m_builder.AddStatement(new PTX::MoveInstruction<PTX::Int64Type>(offsetsRegister, prefixSum));
	}

private:
	ComparisonOperation JoinOperation(const HorseIR::FunctionDeclaration *function)
	{
		if (function->GetKind() == HorseIR::FunctionDeclaration::Kind::Builtin)
		{
			auto builtinFunction = static_cast<const HorseIR::BuiltinFunction *>(function);
			switch (builtinFunction->GetPrimitive())
			{
				case HorseIR::BuiltinFunction::Primitive::Less:
					return ComparisonOperation::Less;
				case HorseIR::BuiltinFunction::Primitive::Greater:
					return ComparisonOperation::Greater;
				case HorseIR::BuiltinFunction::Primitive::LessEqual:
					return ComparisonOperation::LessEqual;
				case HorseIR::BuiltinFunction::Primitive::GreaterEqual:
					return ComparisonOperation::GreaterEqual;
				case HorseIR::BuiltinFunction::Primitive::Equal:
					return ComparisonOperation::Equal;
				case HorseIR::BuiltinFunction::Primitive::NotEqual:
					return ComparisonOperation::NotEqual;
			}
		}
		Generator::Error("comparison function '" + HorseIR::PrettyPrinter::PrettyString(function, true) + "'");
	}
};

}
