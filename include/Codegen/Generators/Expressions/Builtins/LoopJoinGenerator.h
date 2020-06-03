#pragma once

#include "Codegen/Generators/Generator.h"

#include "Codegen/Builder.h"
#include "Codegen/Generators/Data/TargetCellGenerator.h"

#include "HorseIR/Tree/Tree.h"

#include "PTX/PTX.h"

namespace Codegen {

template<PTX::Bits B>
class LoopJoinGenerator : public Generator
{
public:
	using Generator::Generator;

	std::string Name() const override { return "LoopJoinGenerator"; }

	void Generate(const HorseIR::LValue *target, const std::vector<HorseIR::Operand *>& arguments)
	{
		std::vector<HorseIR::Operand *> functionArguments(std::begin(arguments), std::end(arguments) - 4);
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

		InternalFindGenerator<B, PTX::Int64Type> findGenerator(this->m_builder, FindOperation::Indexes, joinOperations);
		findGenerator.Generate(target, dataArguments);
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
