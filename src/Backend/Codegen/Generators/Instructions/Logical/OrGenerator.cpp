#include "Backend/Codegen/Generators/Instructions/Logical/OrGenerator.h"

#include "Backend/Codegen/Generators/Operands/PredicateGenerator.h"

namespace Backend {
namespace Codegen {

void OrGenerator::Generate(const PTX::_OrInstruction *instruction)
{
	instruction->Dispatch(*this);
}

template<class T>
void OrGenerator::Visit(const PTX::OrInstruction<T> *instruction)
{
	// Types:
	//   - Predicate
	//   - Bit16, Bit32, Bit64    
	// Modifiers: --

	if constexpr(std::is_same<T, PTX::PredicateType>::value)
	{
		// Generate operands

		PredicateGenerator predicateGenerator(this->m_builder);
		auto destination = predicateGenerator.Generate(instruction->GetDestination()).first;
		auto [sourceA, sourceA_Not] = predicateGenerator.Generate(instruction->GetSourceA());
		auto [sourceB, sourceB_Not] = predicateGenerator.Generate(instruction->GetSourceB());

		// Flags

		auto flags = SASS::PSETPInstruction::Flags::None;
		if (sourceA_Not)
		{
			flags |= SASS::PSETPInstruction::Flags::NOT_A;
		}
		if (sourceB_Not)
		{
			flags |= SASS::PSETPInstruction::Flags::NOT_B;
		}

		// Generate instruction

		this->AddInstruction(new SASS::PSETPInstruction(
			destination, SASS::PT, sourceA, sourceB, SASS::PT,
			SASS::PSETPInstruction::BooleanOperator1::OR,
			SASS::PSETPInstruction::BooleanOperator2::AND,
			flags
		));
	}
	//TODO: Instruction Or<T> types/modifiers
}

}
}
