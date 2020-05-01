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

		DispatchType(*this, arguments.back()->GetType(), target, joinOperations, dataArguments);

		// Compute prefix sum, getting the offset for each thread

		PrefixSumGenerator<B, PTX::Int64Type> prefixSumGenerator(this->m_builder);
		auto prefixSum = prefixSumGenerator.Generate(m_targetRegister, PrefixSumMode::Exclusive);

		this->m_builder.AddStatement(new PTX::MoveInstruction<PTX::Int64Type>(m_targetRegister, prefixSum));
	}

	template<typename T>
	void GenerateVector(const HorseIR::LValue *target, const std::vector<ComparisonOperation>& joinOperations, const std::vector<HorseIR::Operand *>& dataArguments)
	{
		InternalFindGenerator<B, T, PTX::Int64Type> findGenerator(this->m_builder, FindOperation::Count, joinOperations.at(0));
		m_targetRegister = findGenerator.Generate(target, dataArguments);
	}

	template<typename T>
	void GenerateList(const HorseIR::LValue *target, const std::vector<ComparisonOperation>& functions, const std::vector<HorseIR::Operand *>& arguments)
	{
		//TODO: @join_count list
		// if (this->m_builder.GetInputOptions().IsVectorGeometry())
		// {
		// 	BuiltinGenerator<B, PTX::Int64Type>::Unimplemented("list-in-vector");
		// }

		// Lists are handled by the vector code through a projection

		// GenerateVector<T>(target, arguments);
	}

	template<typename T>
	void GenerateTuple(unsigned int index, const HorseIR::LValue *target, const std::vector<ComparisonOperation>& functions, const std::vector<HorseIR::Operand *>& arguments)
	{
		//TODO: @join_count tuple
		// BuiltinGenerator<B, PTX::Int64Type>::Unimplemented("list-in-vector");
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
