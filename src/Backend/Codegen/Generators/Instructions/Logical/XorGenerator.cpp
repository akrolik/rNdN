#include "Backend/Codegen/Generators/Instructions/Logical/XorGenerator.h"

#include "Backend/Codegen/Generators/Operands/PredicateGenerator.h"

namespace Backend {
namespace Codegen {

void XorGenerator::Generate(const PTX::_XorInstruction *instruction)
{
	instruction->Dispatch(*this);
}

template<class T>
void XorGenerator::Visit(const PTX::XorInstruction<T> *instruction)
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
			SASS::PSETPInstruction::BooleanOperator1::XOR,
			SASS::PSETPInstruction::BooleanOperator2::AND
		));
	}
	//TODO: Instruction Xor<T> types/modifiers
}

}
}
