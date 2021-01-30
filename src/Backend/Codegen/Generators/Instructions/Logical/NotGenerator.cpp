#include "Backend/Codegen/Generators/Instructions/Logical/NotGenerator.h"

#include "Backend/Codegen/Generators/Operands/PredicateGenerator.h"

namespace Backend {
namespace Codegen {

void NotGenerator::Generate(const PTX::_NotInstruction *instruction)
{
	instruction->Dispatch(*this);
}

template<class T>
void NotGenerator::Visit(const PTX::NotInstruction<T> *instruction)
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
		auto [source, source_Not] = predicateGenerator.Generate(instruction->GetSource());

		// Flags

		auto flags = SASS::PSETPInstruction::Flags::NOT_A;
		if (source_Not)
		{
			flags = SASS::PSETPInstruction::Flags::None;
		}

		// Generate instruction

		this->AddInstruction(new SASS::PSETPInstruction(
			destination, SASS::PT, source, SASS::PT, SASS::PT,
			SASS::PSETPInstruction::BooleanOperator1::AND,
			SASS::PSETPInstruction::BooleanOperator2::AND,
			flags
		));
	}
	//TODO: Instruction Not<T> types/modifiers
}

}
}
