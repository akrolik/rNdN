#include "Backend/Codegen/Generators/Instructions/Logical/NotGenerator.h"

#include "Backend/Codegen/Generators/Operands/CompositeGenerator.h"
#include "Backend/Codegen/Generators/Operands/PredicateGenerator.h"
#include "Backend/Codegen/Generators/Operands/RegisterGenerator.h"

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
	else
	{
		RegisterGenerator registerGenerator(this->m_builder);
		auto [destination, destination_Hi] = registerGenerator.Generate(instruction->GetDestination());

		CompositeGenerator compositeGenerator(this->m_builder);
		auto [source, source_Hi] = compositeGenerator.Generate(instruction->GetSource());

		this->AddInstruction(new SASS::LOPInstruction(
			destination, SASS::RZ, source, SASS::LOPInstruction::BooleanOperator::PASS_B,
			SASS::LOPInstruction::Flags::INV
		));

		if constexpr(T::TypeBits == PTX::Bits::Bits64)
		{
			this->AddInstruction(new SASS::LOPInstruction(
				destination_Hi, SASS::RZ, source_Hi, SASS::LOPInstruction::BooleanOperator::PASS_B,
				SASS::LOPInstruction::Flags::INV
			));
		}
	}
}

}
}
