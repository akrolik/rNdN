#pragma once

#include "HorseIR/Traversal/ConstVisitor.h"
#include "Codegen/Generators/Expressions/Builtins/BuiltinGenerator.h"

#include "Codegen/Builder.h"
#include "Codegen/Generators/Data/ValueLoadGenerator.h"
#include "Codegen/Generators/Expressions/OperandGenerator.h"

#include "HorseIR/Tree/Tree.h"

#include "PTX/PTX.h"

namespace Codegen {

template<PTX::Bits B, class T>
class IndexedReadGenerator : public BuiltinGenerator<B, T>, public HorseIR::ConstVisitor
{
public:
	using BuiltinGenerator<B, T>::BuiltinGenerator;

	const PTX::Register<PTX::PredicateType> *GenerateCompressionPredicate(const std::vector<HorseIR::Operand *>& arguments) override
	{
		return OperandCompressionGenerator::BinaryCompressionRegister(this->m_builder, arguments);
	}

	const PTX::Register<T> *Generate(const HorseIR::LValue *target, const std::vector<HorseIR::Operand *>& arguments) override
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
		auto resources = this->m_builder.GetLocalResources();
		if (auto predicate = resources->template GetCompressedRegister<T>(m_targetRegister))
		{
			auto label = this->m_builder.CreateLabel("END");
			this->m_builder.AddStatement(new PTX::BranchInstruction(label, predicate, true));
			this->m_builder.AddStatement(new PTX::BlankStatement());

			auto name = NameUtils::VariableName(identifier);
			ValueLoadGenerator<B> loadGenerator(this->m_builder);
			loadGenerator.template GeneratePointer<T>(name, m_targetRegister, m_index);

			this->m_builder.AddStatement(new PTX::BlankStatement());
			this->m_builder.AddStatement(label);
		}
		else
		{
			auto name = NameUtils::VariableName(identifier);
			ValueLoadGenerator<B> loadGenerator(this->m_builder);
			loadGenerator.template GeneratePointer<T>(name, m_targetRegister, m_index);
		}
	}

private:
	const PTX::TypedOperand<PTX::UInt32Type> *m_index = nullptr;
	const PTX::Register<T> *m_targetRegister = nullptr;
};

}
