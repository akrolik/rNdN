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
class ToListGenerator : public BuiltinGenerator<B, T>
{
public:
	using BuiltinGenerator<B, T>::BuiltinGenerator;

	std::string Name() const override { return "ToListGenerator"; }

	// The output of the tolist function has no compression predicate. We therefore do not implement GenerateCompressionPredicate in this subclass

	PTX::Register<T> *Generate(const HorseIR::LValue *target, const std::vector<const HorseIR::Operand *>& arguments) override
	{
		OperandGenerator<B, T> opGen(this->m_builder);
		auto value = opGen.GenerateRegister(arguments.at(0), OperandGenerator<B, T>::LoadKind::ListData);

		auto targetRegister = this->GenerateTargetRegister(target, arguments);

		MoveGenerator<T> moveGenerator(this->m_builder);
		moveGenerator.Generate(targetRegister, value);

		return targetRegister;
	}
};

}
}
