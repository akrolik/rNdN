#include "Backend/Codegen/Generators/Instructions/Logical/AndGenerator.h"

#include "Backend/Codegen/Generators/Operands/CompositeGenerator.h"
#include "Backend/Codegen/Generators/Operands/PredicateGenerator.h"
#include "Backend/Codegen/Generators/Operands/RegisterGenerator.h"

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
			SASS::PSETPInstruction::BooleanOperator1::AND,
			SASS::PSETPInstruction::BooleanOperator2::AND,
			flags
		));
	}
	else
	{
		RegisterGenerator registerGenerator(this->m_builder);
		auto [destination, destination_Hi] = registerGenerator.Generate(instruction->GetDestination());
		auto [sourceA, sourceA_Hi] = registerGenerator.Generate(instruction->GetSourceA());

		CompositeGenerator compositeGenerator(this->m_builder);
		auto [sourceB, sourceB_Hi] = compositeGenerator.Generate(instruction->GetSourceB());

		this->AddInstruction(new SASS::LOPInstruction(
			destination, sourceA, sourceB, SASS::LOPInstruction::BooleanOperator::AND
		));

		if constexpr(T::TypeBits == PTX::Bits::Bits64)
		{
			this->AddInstruction(new SASS::LOPInstruction(
				destination_Hi, sourceA_Hi, sourceB_Hi, SASS::LOPInstruction::BooleanOperator::AND
			));
		}
	}
}

}
}
