#pragma once

#include "Codegen/Generators/Generator.h"

#include "Codegen/Builder.h"
#include "Codegen/Generators/Expressions/Builtins/InternalFindGenerator.h"
#include "Codegen/Generators/Indexing/PrefixSumGenerator.h"

#include "HorseIR/Tree/Tree.h"
#include "HorseIR/Utils/PrettyPrinter.h"
#include "HorseIR/Utils/TypeUtils.h"

#include "PTX/PTX.h"

namespace Codegen {

template<PTX::Bits B>
class JoinCountGenerator : public Generator
{
public:
	using Generator::Generator;

	std::string Name() const override { return "JoinCountGenerator"; }

	void Generate(const HorseIR::LValue *target, const std::vector<HorseIR::Operand *>& arguments)
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
		m_targetRegister = findGenerator.Generate(target, dataArguments);

		// Compute prefix sum, getting the offset for each thread

		PrefixSumGenerator<B, PTX::Int64Type> prefixSumGenerator(this->m_builder);
		auto prefixSum = prefixSumGenerator.Generate(m_targetRegister, PrefixSumMode::Exclusive);

		this->m_builder.AddStatement(new PTX::MoveInstruction<PTX::Int64Type>(m_targetRegister, prefixSum));
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

	const PTX::Register<PTX::Int64Type> *m_targetRegister = nullptr;
};

}
