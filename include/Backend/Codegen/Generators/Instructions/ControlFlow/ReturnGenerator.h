#pragma once

#include "Backend/Codegen/Generators/Instructions/PredicatedInstructionGenerator.h"

#include "PTX/Tree/Tree.h"
#include "SASS/Tree/Tree.h"

namespace Backend {
namespace Codegen {

class ReturnGenerator : public PredicatedInstructionGenerator
{
public:
	using PredicatedInstructionGenerator::PredicatedInstructionGenerator;

	std::string Name() const override { return "ReturnGenerator"; }

	// Generators

	void Generate(const PTX::ReturnInstruction *instruction);

	void GenerateMaxwell(const PTX::ReturnInstruction *instruction);
	void GenerateVolta(const PTX::ReturnInstruction *instruction);
};

}
}
