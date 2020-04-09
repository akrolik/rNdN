#pragma once

#include "Codegen/Generators/Expressions/Builtins/BuiltinGenerator.h"

#include "Codegen/Builder.h"
#include "Codegen/Generators/Expressions/MoveGenerator.h"
#include "Codegen/Generators/Expressions/OperandGenerator.h"

#include "HorseIR/Tree/Tree.h"

#include "PTX/PTX.h"

namespace Codegen {

template<PTX::Bits B, class T>
class VectorGenerator : public BuiltinGenerator<B, T>
{
public:
	using BuiltinGenerator<B, T>::BuiltinGenerator;

	std::string Name() const override { return "VectorGenerator"; }

	// The output of the vector function has no compression predicate. We therefore do not implement GenerateCompressionPredicate in this subclass

	const PTX::Register<T> *Generate(const HorseIR::LValue *target, const std::vector<HorseIR::Operand *>& arguments) override
	{
		OperandGenerator<B, T> opGen(this->m_builder);
		auto src = opGen.GenerateOperand(arguments.at(1), OperandGenerator<B, T>::LoadKind::Vector);

		auto targetRegister = this->GenerateTargetRegister(target, arguments);

		MoveGenerator<T> moveGenerator(this->m_builder);
		moveGenerator.Generate(targetRegister, src);

		return targetRegister;
	}
};

}
