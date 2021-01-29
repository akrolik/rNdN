#pragma once

#include "Backend/Codegen/Generators/Instructions/PredicatedInstructionGenerator.h"

#include "PTX/Tree/Tree.h"
#include "SASS/SASS.h"

namespace Backend {
namespace Codegen {

class MoveGenerator : public PredicatedInstructionGenerator
{
public:
	using PredicatedInstructionGenerator::PredicatedInstructionGenerator;

	std::string Name() const override { return "MoveGenerator"; }

	// Instruction

	void Generate(const PTX::_MoveInstruction *instruction);

	template<class T>
	void Visit(const PTX::MoveInstruction<T> *instruction);
};

}
}
