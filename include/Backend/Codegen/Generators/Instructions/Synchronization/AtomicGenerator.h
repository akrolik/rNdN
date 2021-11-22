#pragma once

#include "Backend/Codegen/Generators/Instructions/PredicatedInstructionGenerator.h"

#include "PTX/Tree/Tree.h"
#include "SASS/Tree/Tree.h"

namespace Backend {
namespace Codegen {

class AtomicGenerator : public PredicatedInstructionGenerator
{
public:
	using PredicatedInstructionGenerator::PredicatedInstructionGenerator;

	std::string Name() const override { return "AtomicGenerator"; }

	// Instruction

	void Generate(const PTX::_AtomicInstruction *instruction);

	template<PTX::Bits B, class T, class S>
	void Visit(const PTX::AtomicInstruction<B, T, S> *instruction);
	
	template<PTX::Bits B, class T, class S>
	void GenerateMaxwell(const PTX::AtomicInstruction<B, T, S> *instruction);

	template<PTX::Bits B, class T, class S>
	void GenerateVolta(const PTX::AtomicInstruction<B, T, S> *instruction);

private:
	template<class I, PTX::Bits B, class T, class S>
	typename I::Type InstructionType(const PTX::AtomicInstruction<B, T, S> *instruction);

	template<class I, PTX::Bits B, class T, class S>
	typename I::Mode InstructionMode(const PTX::AtomicInstruction<B, T, S> *instruction);
};

}
}
