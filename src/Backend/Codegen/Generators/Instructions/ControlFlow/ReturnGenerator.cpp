#include "Backend/Codegen/Generators/Instructions/ControlFlow/ReturnGenerator.h"

#include "Backend/Codegen/Generators/ArchitectureDispatch.h"

namespace Backend {
namespace Codegen {

void ReturnGenerator::Generate(const PTX::ReturnInstruction *instruction)
{
	// Instruction predicate

	this->SetPredicatedInstruction(instruction);

	ArchitectureDispatch::Dispatch(*this, instruction);
}

void ReturnGenerator::GenerateMaxwell(const PTX::ReturnInstruction *instruction)
{
	this->AddInstruction(new SASS::Maxwell::EXITInstruction());
}

void ReturnGenerator::GenerateVolta(const PTX::ReturnInstruction *instruction)
{
	this->AddInstruction(new SASS::Volta::EXITInstruction());
}

}
}

