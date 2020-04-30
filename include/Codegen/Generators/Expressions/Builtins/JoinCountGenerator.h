#pragma once

#include "Codegen/Generators/Generator.h"

#include "Codegen/Builder.h"
#include "Codegen/Generators/Expressions/Builtins/InternalFindGenerator.h"
#include "Codegen/Generators/Indexing/PrefixSumGenerator.h"

#include "HorseIR/Tree/Tree.h"

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
		std::vector<HorseIR::Operand *> dataArguments;
		dataArguments.push_back(arguments.at(arguments.size() - 2));
		dataArguments.push_back(arguments.at(arguments.size() - 1));

		DispatchType(*this, arguments.back()->GetType(), target, dataArguments);

		// Compute prefix sum, getting the offset for each thread

		PrefixSumGenerator<B, PTX::Int64Type> prefixSumGenerator(this->m_builder);
		auto prefixSum = prefixSumGenerator.Generate(m_targetRegister, PrefixSumMode::Exclusive);

		this->m_builder.AddStatement(new PTX::MoveInstruction<PTX::Int64Type>(m_targetRegister, prefixSum));
	}

	template<typename T>
	void GenerateVector(const HorseIR::LValue *target, const std::vector<HorseIR::Operand *>& arguments)
	{
		InternalFindGenerator<B, T, PTX::Int64Type> findGenerator(this->m_builder, FindOperation::Count);
		m_targetRegister = findGenerator.Generate(target, arguments);
	}

	template<typename T>
	void GenerateList(const HorseIR::LValue *target, const std::vector<HorseIR::Operand *>& arguments)
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
	void GenerateTuple(unsigned int index, const HorseIR::LValue *target, const std::vector<HorseIR::Operand *>& arguments)
	{
		//TODO: @join_count tuple
		// BuiltinGenerator<B, PTX::Int64Type>::Unimplemented("list-in-vector");
	}

private:
	const PTX::Register<PTX::Int64Type> *m_targetRegister = nullptr;
};

}
