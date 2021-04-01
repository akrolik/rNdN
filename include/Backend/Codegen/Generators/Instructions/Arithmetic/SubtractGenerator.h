#pragma once

#include "Backend/Codegen/Generators/Instructions/PredicatedInstructionGenerator.h"

#include "PTX/Tree/Tree.h"
#include "SASS/SASS.h"

namespace Backend {
namespace Codegen {

class SubtractGenerator : public PredicatedInstructionGenerator
{
public:
	using PredicatedInstructionGenerator::PredicatedInstructionGenerator;

	std::string Name() const override { return "SubtractGenerator"; }

	// Instruction

	void Generate(const PTX::_SubtractInstruction *instruction);

	template<class T>
	void Visit(const PTX::SubtractInstruction<T> *instruction);
};

}
}
