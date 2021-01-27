#pragma once

#include "Backend/Codegen/Generators/Instructions/PredicatedInstructionGenerator.h"

#include "PTX/Tree/Tree.h"
#include "SASS/SASS.h"

namespace Backend {
namespace Codegen {

class LoadGenerator : public PredicatedInstructionGenerator
{
public:
	using PredicatedInstructionGenerator::PredicatedInstructionGenerator;

	std::string Name() const override { return "LoadGenerator"; }

	// Instruction

	void Generate(const PTX::_LoadInstruction *instruction);

	template<PTX::Bits B, class T, class S, PTX::LoadSynchronization A>
	void Visit(const PTX::LoadInstruction<B, T, S, A> *instruction);

private:
	template<class T>
	SASS::LDGInstruction::Type InstructionType();
};

}
}
