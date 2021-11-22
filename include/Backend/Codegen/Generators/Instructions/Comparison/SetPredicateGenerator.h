#pragma once

#include "Backend/Codegen/Generators/Instructions/PredicatedInstructionGenerator.h"

#include "PTX/Tree/Tree.h"
#include "SASS/Tree/Tree.h"

namespace Backend {
namespace Codegen {

class SetPredicateGenerator : public PredicatedInstructionGenerator
{
public:
	using PredicatedInstructionGenerator::PredicatedInstructionGenerator;

	std::string Name() const override { return "SetPredicateGenerator"; }

	// Instruction

	void Generate(const PTX::_SetPredicateInstruction *instruction);

	template<class T>
	void Visit(const PTX::SetPredicateInstruction<T> *instruction);

	template<class T>
	void GenerateMaxwell(const PTX::SetPredicateInstruction<T> *instruction);

	template<class T>
	void GenerateVolta(const PTX::SetPredicateInstruction<T> *instruction);

private:
	template<class I, class T>
	typename I::ComparisonOperator IInstructionComparisonOperator(const PTX::SetPredicateInstruction<T> *instruction);

	template<class I, class T>
	typename I::BooleanOperator IInstructionBooleanOperator(const PTX::SetPredicateInstruction<T> *instruction);

	template<class I, class T>
	typename I::ComparisonOperator DInstructionComparisonOperator(const PTX::SetPredicateInstruction<T> *instruction);

	template<class I, class T>
	typename I::BooleanOperator DInstructionBooleanOperator(const PTX::SetPredicateInstruction<T> *instruction);
};

}
}
