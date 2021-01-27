#pragma once

#include "Backend/Codegen/Generators/Instructions/PredicatedInstructionGenerator.h"

#include "PTX/Tree/Tree.h"
#include "SASS/SASS.h"

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
};

}
}
