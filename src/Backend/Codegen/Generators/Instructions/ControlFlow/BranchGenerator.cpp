#include "Backend/Codegen/Generators/Instructions/ControlFlow/BranchGenerator.h"

namespace Backend {
namespace Codegen {

void BranchGenerator::Generate(const PTX::BranchInstruction *instruction)
{
	//TODO: Predicated
	auto name = instruction->GetLabel()->GetName();
	if (instruction->GetUniform())
	{
		this->m_builder.AddInstruction(new SASS::BRAInstruction(name, SASS::BRAInstruction::Flags::U));
	}
	else
	{
		this->m_builder.AddInstruction(new SASS::BRAInstruction(name));
	}
}

}
}

