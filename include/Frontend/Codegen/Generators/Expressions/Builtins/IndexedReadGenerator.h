#pragma once

#include "HorseIR/Traversal/ConstVisitor.h"
#include "Frontend/Codegen/Generators/Expressions/Builtins/BuiltinGenerator.h"

#include "Frontend/Codegen/Builder.h"
#include "Frontend/Codegen/Generators/Expressions/OperandGenerator.h"

#include "HorseIR/Tree/Tree.h"

#include "PTX/Tree/Tree.h"

namespace Frontend {
namespace Codegen {

template<PTX::Bits B, class T>
class IndexedReadGenerator : public BuiltinGenerator<B, T>, public HorseIR::ConstVisitor
{
public:
	using BuiltinGenerator<B, T>::BuiltinGenerator;

	std::string Name() const override { return "IndexedReadGenerator"; }

	PTX::Register<PTX::PredicateType> *GenerateCompressionPredicate(const std::vector<const HorseIR::Operand *>& arguments) override
	{
		return OperandCompressionGenerator::BinaryCompressionRegister(this->m_builder, arguments);
	}

	PTX::Register<T> *Generate(const HorseIR::LValue *target, const std::vector<const HorseIR::Operand *>& arguments) override
	{
		// Value in argument 0, index in argument 1

		OperandGenerator<B, PTX::UInt32Type> opGen(this->m_builder);
		m_index = opGen.GenerateOperand(arguments.at(1), OperandGenerator<B, PTX::UInt32Type>::LoadKind::Vector);
		m_targetRegister = this->GenerateTargetRegister(target, arguments);

		arguments.at(0)->Accept(*this);

		return m_targetRegister;
	}

	void Visit(const HorseIR::Literal *literal) override
	{
		BuiltinGenerator<B, T>::Unimplemented("indexed read of literal result");
	}

	void Visit(const HorseIR::Identifier *identifier) override
	{
		OperandGenerator<B, T> opGen(this->m_builder);
		auto value = opGen.GenerateOperand(identifier, m_index, this->m_builder.UniqueIdentifier("index"));

		MoveGenerator<T> moveGenerator(this->m_builder);
		moveGenerator.Generate(m_targetRegister, value);
	}

private:
	PTX::TypedOperand<PTX::UInt32Type> *m_index = nullptr;
	PTX::Register<T> *m_targetRegister = nullptr;
};

}
}
