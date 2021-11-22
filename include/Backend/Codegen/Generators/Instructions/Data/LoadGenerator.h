#pragma once

#include "Backend/Codegen/Generators/Instructions/PredicatedInstructionGenerator.h"

#include "PTX/Tree/Tree.h"
#include "SASS/Tree/Tree.h"

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

	template<PTX::Bits B, class T, class S, PTX::LoadSynchronization A>
	void GenerateMaxwell(const PTX::LoadInstruction<B, T, S, A> *instruction);

	template<PTX::Bits B, class T, class S, PTX::LoadSynchronization A>
	void GenerateVolta(const PTX::LoadInstruction<B, T, S, A> *instruction);

private:
	template<typename I, class T>
	I InstructionType();
};

}
}
