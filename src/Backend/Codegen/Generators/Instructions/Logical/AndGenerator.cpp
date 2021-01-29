#include "Backend/Codegen/Generators/Instructions/Logical/AndGenerator.h"

#include "Backend/Codegen/Generators/Operands/PredicateGenerator.h"

namespace Backend {
namespace Codegen {

void AndGenerator::Generate(const PTX::_AndInstruction *instruction)
{
	instruction->Dispatch(*this);
}

template<class T>
void AndGenerator::Visit(const PTX::AndInstruction<T> *instruction)
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
			SASS::PSETPInstruction::BooleanOperator1::AND,
			SASS::PSETPInstruction::BooleanOperator2::AND
		));
	}
	//TODO: Instruction And<T> types (LOP)
}

}
}
