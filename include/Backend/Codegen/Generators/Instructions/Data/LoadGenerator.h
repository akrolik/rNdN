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
	template<typename LDInstruction, class T>
	typename LDInstruction::Type InstructionType();

	template<class MOVInstruction, class LDGInstruction, class LDSInstruction, PTX::Bits B, class T, class S, PTX::LoadSynchronization A>
	void GenerateInstruction(const PTX::LoadInstruction<B, T, S, A> *instruction);
};

}
}
