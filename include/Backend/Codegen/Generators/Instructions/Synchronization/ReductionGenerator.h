#pragma once

#include "Backend/Codegen/Generators/Instructions/PredicatedInstructionGenerator.h"

#include "PTX/Tree/Tree.h"
#include "SASS/SASS.h"

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

private:
	template<PTX::Bits B, class T, class S>
	SASS::REDInstruction::Type InstructionType(const PTX::ReductionInstruction<B, T, S> *instruction);

	template<PTX::Bits B, class T, class S>
	SASS::REDInstruction::Mode InstructionMode(const PTX::ReductionInstruction<B, T, S> *instruction);
};

}
}
