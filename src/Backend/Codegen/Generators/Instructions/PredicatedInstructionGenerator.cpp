#include "Backend/Codegen/Generators/Instructions/PredicatedInstructionGenerator.h"

#include "Backend/Codegen/Generators/Operands/PredicateGenerator.h"

namespace Backend {
namespace Codegen {

void PredicatedInstructionGenerator::SetPredicatedInstruction(const PTX::PredicatedInstruction *instruction)
{
	if (const auto& [predicate, negate] = instruction->GetPredicate(); predicate != nullptr)
	{
		PredicateGenerator predicateGenerator(this->m_builder);
		m_predicate = predicateGenerator.Generate(predicate);
		m_negatePredicate = negate;
	}
}

void PredicatedInstructionGenerator::AddInstruction(SASS::PredicatedInstruction *instruction)
{
	instruction->SetPredicate(m_predicate, m_negatePredicate);
	this->m_builder.AddInstruction(instruction);
}

}
}
