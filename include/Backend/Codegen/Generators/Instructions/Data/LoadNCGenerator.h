#pragma once

#include "Backend/Codegen/Generators/Instructions/PredicatedInstructionGenerator.h"

#include "PTX/Tree/Tree.h"
#include "SASS/Tree/Tree.h"

namespace Backend {
namespace Codegen {

class LoadNCGenerator : public PredicatedInstructionGenerator
{
public:
	using PredicatedInstructionGenerator::PredicatedInstructionGenerator;

	std::string Name() const override { return "LoadNCGenerator"; }

	// Instruction

	void Generate(const PTX::_LoadNCInstruction *instruction);

	template<PTX::Bits B, class T, class S>
	void Visit(const PTX::LoadNCInstruction<B, T, S> *instruction);

private:
	template<class T>
	SASS::LDGInstruction::Type InstructionType();
};

}
}
