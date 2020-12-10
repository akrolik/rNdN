#pragma once

#include "Frontend/Codegen/Generators/Expressions/Builtins/BuiltinGenerator.h"

#include "Frontend/Codegen/Builder.h"
#include "Frontend/Codegen/Generators/Expressions/MoveGenerator.h"
#include "Frontend/Codegen/Generators/Expressions/OperandGenerator.h"

#include "HorseIR/Tree/Tree.h"

#include "PTX/Tree/Tree.h"

namespace Frontend {
namespace Codegen {

template<PTX::Bits B, class T>
class IndexedWriteGenerator : public BuiltinGenerator<B, T>
{
public:
	using BuiltinGenerator<B, T>::BuiltinGenerator;

	std::string Name() const override { return "IndexedWriteGenerator"; }

	PTX::Register<PTX::PredicateType> *GenerateCompressionPredicate(const std::vector<HorseIR::Operand *>& arguments) override
	{
		std::vector<HorseIR::Operand *> l_arguments;
		l_arguments.insert(std::begin(l_arguments), std::begin(arguments) + 1, std::end(arguments));
		return OperandCompressionGenerator::BinaryCompressionRegister(this->m_builder, l_arguments);
	}

	PTX::Register<T> *Generate(const HorseIR::LValue *target, const std::vector<HorseIR::Operand *>& arguments) override
	{
		// Value in argument 0, index in argument 1

		OperandGenerator<B, PTX::UInt32Type> opIndexGen(this->m_builder);
		auto index = opIndexGen.GenerateOperand(arguments.at(1), OperandGenerator<B, PTX::UInt32Type>::LoadKind::Vector);

		OperandGenerator<B, T> opGen(this->m_builder);
		auto value = opGen.GenerateOperand(arguments.at(2), OperandGenerator<B, T>::LoadKind::Vector);

		// Generate copies of the data and index for writing

		auto resources = this->m_builder.GetLocalResources();
		auto writeIndex = resources->template AllocateTemporary<PTX::UInt32Type>();
		this->m_builder.AddStatement(new PTX::MoveInstruction<PTX::UInt32Type>(writeIndex, index));

		auto targetRegister = this->GenerateTargetRegister(target, arguments);
		MoveGenerator<T> moveGenerator(this->m_builder);
		moveGenerator.Generate(targetRegister, value);

		// Set indexing used for writing output data

		resources->SetIndexedRegister(targetRegister, writeIndex);

		return targetRegister;
	}
};

}
}
