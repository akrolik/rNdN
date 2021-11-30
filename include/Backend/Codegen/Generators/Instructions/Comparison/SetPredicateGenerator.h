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
	template<class ISETPInstruction, class T>
	typename ISETPInstruction::ComparisonOperator IInstructionComparisonOperator(const PTX::SetPredicateInstruction<T> *instruction);

	template<class DSETPInstruction, class T>
	typename DSETPInstruction::ComparisonOperator DInstructionComparisonOperator(const PTX::SetPredicateInstruction<T> *instruction);

	template<class SETPInstruction, class T>
	typename SETPInstruction::BooleanOperator InstructionBooleanOperator(const PTX::SetPredicateInstruction<T> *instruction);

	template<class ISETPInstruction, class DSETPInstruction, class PRMTInstruction, class T>
	void GenerateInstruction(const PTX::SetPredicateInstruction<T> *instruction);
};

}
}
