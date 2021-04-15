#pragma once

#include "Backend/Codegen/Generators/Instructions/PredicatedInstructionGenerator.h"

#include "PTX/Tree/Tree.h"
#include "SASS/Tree/Tree.h"

namespace Backend {
namespace Codegen {

class StoreGenerator : public PredicatedInstructionGenerator
{
public:
	using PredicatedInstructionGenerator::PredicatedInstructionGenerator;

	std::string Name() const override { return "StoreGenerator"; }

	// Instruction

	void Generate(const PTX::_StoreInstruction *instruction);

	template<PTX::Bits B, class T, class S, PTX::StoreSynchronization A>
	void Visit(const PTX::StoreInstruction<B, T, S, A> *instruction);

private:
	template<typename I, class T>
	I InstructionType();
};

}
}
