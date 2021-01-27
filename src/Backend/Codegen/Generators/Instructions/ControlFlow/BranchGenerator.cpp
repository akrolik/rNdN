#include "Backend/Codegen/Generators/Instructions/ControlFlow/BranchGenerator.h"

namespace Backend {
namespace Codegen {

void BranchGenerator::Generate(const PTX::BranchInstruction *instruction)
{
	// Predicated instruction

	//TODO: Branches unsupported
	return;
	this->SetPredicatedInstruction(instruction);

	// Generate branch instruction

	auto name = instruction->GetLabel()->GetName();
	if (instruction->GetUniform())
	{
		this->AddInstruction(new SASS::BRAInstruction(name, SASS::BRAInstruction::Flags::U));
	}
	else
	{
		this->AddInstruction(new SASS::BRAInstruction(name));
	}
}

}
}

