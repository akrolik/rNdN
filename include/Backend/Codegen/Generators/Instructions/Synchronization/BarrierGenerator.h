#pragma once

#include "Backend/Codegen/Generators/Instructions/PredicatedInstructionGenerator.h"

#include "PTX/Tree/Tree.h"
#include "SASS/Tree/Tree.h"

namespace Backend {
namespace Codegen {

class BarrierGenerator : public PredicatedInstructionGenerator
{
public:
	using PredicatedInstructionGenerator::PredicatedInstructionGenerator;

	std::string Name() const override { return "BarrierGenerator"; }

	// Instruction

	void Generate(const PTX::BarrierInstruction *instruction);

	void GenerateMaxwell(const PTX::BarrierInstruction *instruction);
	void GenerateVolta(const PTX::BarrierInstruction *instruction);
};

}
}
