#pragma once

#include "Backend/Codegen/Generators/Instructions/PredicatedInstructionGenerator.h"

#include "PTX/Tree/Tree.h"
#include "SASS/Tree/Tree.h"

namespace Backend {
namespace Codegen {

class ReductionGenerator : public PredicatedInstructionGenerator
{
public:
	using PredicatedInstructionGenerator::PredicatedInstructionGenerator;

	std::string Name() const override { return "ReductionGenerator"; }

	// Instruction

	void Generate(const PTX::_ReductionInstruction *instruction);

	template<PTX::Bits B, class T, class S>
	void Visit(const PTX::ReductionInstruction<B, T, S> *instruction);

	template<class REDInstruction, PTX::Bits B, class T, class S>
	void GenerateInstruction(const PTX::ReductionInstruction<B, T, S> *instruction);

private:
	template<class REDInstruction, PTX::Bits B, class T, class S>
	typename REDInstruction::Type InstructionType(const PTX::ReductionInstruction<B, T, S> *instruction);

	template<class REDInstruction, PTX::Bits B, class T, class S>
	typename REDInstruction::Mode InstructionMode(const PTX::ReductionInstruction<B, T, S> *instruction);
};

}
}
