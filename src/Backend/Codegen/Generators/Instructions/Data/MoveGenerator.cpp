#include "Backend/Codegen/Generators/Instructions/Data/MoveGenerator.h"

#include "Backend/Codegen/Generators/Operands/CompositeGenerator.h"
#include "Backend/Codegen/Generators/Operands/RegisterGenerator.h"
#include "Backend/Codegen/Generators/Operands/PredicateGenerator.h"

namespace Backend {
namespace Codegen {

void MoveGenerator::Generate(const PTX::_MoveInstruction *instruction)
{
	instruction->Dispatch(*this);
}

template<class T>
void MoveGenerator::Visit(const PTX::MoveInstruction<T> *instruction)
{
	// Instruction predicate

	this->SetPredicatedInstruction(instruction);

	// Types:
	//   - Predicate
	//   - Bit16, Bit32, Bit64
	//   - Int16, Int32, Int64
	//   - UInt16, UInt32, UInt64
	//   - Float32, Float64
	// Modifiers: --

	if constexpr(std::is_same<T, PTX::PredicateType>::value)
	{
		// Generate operands

		PredicateGenerator predicateGenerator(this->m_builder);
		auto destination = predicateGenerator.Generate(instruction->GetDestination()).first;
		auto [source, source_Not] = predicateGenerator.Generate(instruction->GetSource());

		// Flags

		auto flags = SASS::PSETPInstruction::Flags::None;
		if (source_Not)
		{
			flags |= SASS::PSETPInstruction::Flags::NOT_A;
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
		// Generate operands

		RegisterGenerator registerGenerator(this->m_builder);
		auto [destination_Lo, destination_Hi] = registerGenerator.GeneratePair(instruction->GetDestination());

		CompositeGenerator compositeGenerator(this->m_builder);
		auto [source_Lo, source_Hi] = compositeGenerator.GeneratePair(instruction->GetSource());

		// Generate instruction (no overlap unless equal)

		this->AddInstruction(new SASS::MOVInstruction(destination_Lo, source_Lo));
		if constexpr(T::TypeBits == PTX::Bits::Bits64)
		{
			this->AddInstruction(new SASS::MOVInstruction(destination_Hi, source_Hi));
		}
	}
}

}
}
