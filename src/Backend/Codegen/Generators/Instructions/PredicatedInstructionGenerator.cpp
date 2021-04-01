#include "Backend/Codegen/Generators/Instructions/PredicatedInstructionGenerator.h"

#include "Backend/Codegen/Generators/Operands/PredicateGenerator.h"

namespace Backend {
namespace Codegen {

void PredicatedInstructionGenerator::SetPredicatedInstruction(const PTX::PredicatedInstruction *instruction)
{
	if (const auto& [predicate, negate] = instruction->GetPredicate(); predicate != nullptr)
	{
		PredicateGenerator predicateGenerator(this->m_builder);
		auto [predicateReg, predicate_Not] = predicateGenerator.Generate(predicate);

		m_predicate = predicateReg;
		m_negatePredicate = negate ^ predicate_Not;
	}
	else
	{
		m_predicate = nullptr;
		m_negatePredicate = false;
	}

	m_instruction = instruction;
}

void PredicatedInstructionGenerator::AddInstruction(SASS::PredicatedInstruction *instruction, SASS::Predicate *predicate, bool negatePredicate)
{
	if (predicate != nullptr && m_predicate != nullptr)
	{
		Error(m_instruction, "internal implementation requires predicate");
	}

	if (m_predicate != nullptr)
	{
		instruction->SetPredicate(m_predicate, m_negatePredicate);
	}
	if (predicate != nullptr)
	{
		instruction->SetPredicate(predicate, negatePredicate);
	}

	this->m_builder.AddInstruction(instruction);
}

}
}
