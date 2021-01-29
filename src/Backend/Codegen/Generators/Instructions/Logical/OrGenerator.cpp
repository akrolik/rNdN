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
		auto destination = predicateGenerator.Generate(instruction->GetDestination());
		auto sourceA = predicateGenerator.Generate(instruction->GetSourceA());
		auto sourceB = predicateGenerator.Generate(instruction->GetSourceB());

		// Generate instruction

		this->AddInstruction(new SASS::PSETPInstruction(
			destination, SASS::PT, sourceA, sourceB, SASS::PT,
			SASS::PSETPInstruction::BooleanOperator1::OR,
			SASS::PSETPInstruction::BooleanOperator2::AND
		));
	}
	//TODO: Instruction Or<T> types/modifiers
}

}
}
