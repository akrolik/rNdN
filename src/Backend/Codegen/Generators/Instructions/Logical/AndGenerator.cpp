#include "Backend/Codegen/Generators/Instructions/Logical/AndGenerator.h"

#include "Backend/Codegen/Generators/Operands/CompositeGenerator.h"
#include "Backend/Codegen/Generators/Operands/PredicateGenerator.h"
#include "Backend/Codegen/Generators/Operands/RegisterGenerator.h"

#include "Backend/Codegen/Generators/ArchitectureDispatch.h"

namespace Backend {
namespace Codegen {

void AndGenerator::Generate(const PTX::_AndInstruction *instruction)
{
	instruction->Dispatch(*this);
}

template<class T>
void AndGenerator::Visit(const PTX::AndInstruction<T> *instruction)
{
	// Instruction predicate

	this->SetPredicatedInstruction(instruction);

	// Types:
	//   - Predicate
	//   - Bit16, Bit32, Bit64    
	// Modifiers: --

	ArchitectureDispatch::Dispatch(*this, instruction);
}

template<class T>
void AndGenerator::GenerateMaxwell(const PTX::AndInstruction<T> *instruction)
{
	if constexpr(std::is_same<T, PTX::PredicateType>::value)
	{
		// Generate operands

		PredicateGenerator predicateGenerator(this->m_builder);
		auto destination = predicateGenerator.Generate(instruction->GetDestination()).first;
		auto [sourceA, sourceA_Not] = predicateGenerator.Generate(instruction->GetSourceA());
		auto [sourceB, sourceB_Not] = predicateGenerator.Generate(instruction->GetSourceB());

		// Flags

		auto flags = SASS::Maxwell::PSETPInstruction::Flags::None;
		if (sourceA_Not)
		{
			flags |= SASS::Maxwell::PSETPInstruction::Flags::NOT_A;
		}
		if (sourceB_Not)
		{
			flags |= SASS::Maxwell::PSETPInstruction::Flags::NOT_B;
		}

		// Generate instruction

		this->AddInstruction(new SASS::Maxwell::PSETPInstruction(
			destination, SASS::PT, sourceA, sourceB, SASS::PT,
			SASS::Maxwell::PSETPInstruction::BooleanOperator1::AND,
			SASS::Maxwell::PSETPInstruction::BooleanOperator2::AND,
			flags
		));
	}
	else
	{
		RegisterGenerator registerGenerator(this->m_builder);
		auto [destination_Lo, destination_Hi] = registerGenerator.GeneratePair(instruction->GetDestination());
		auto [sourceA_Lo, sourceA_Hi] = registerGenerator.GeneratePair(instruction->GetSourceA());

		CompositeGenerator compositeGenerator(this->m_builder);
		auto [sourceB_Lo, sourceB_Hi] = compositeGenerator.GeneratePair(instruction->GetSourceB());

		this->AddInstruction(new SASS::Maxwell::LOPInstruction(
			destination_Lo, sourceA_Lo, sourceB_Lo, SASS::Maxwell::LOPInstruction::BooleanOperator::AND
		));

		if constexpr(T::TypeBits == PTX::Bits::Bits64)
		{
			this->AddInstruction(new SASS::Maxwell::LOPInstruction(
				destination_Hi, sourceA_Hi, sourceB_Hi, SASS::Maxwell::LOPInstruction::BooleanOperator::AND
			));
		}
	}
}

template<class T>
void AndGenerator::GenerateVolta(const PTX::AndInstruction<T> *instruction)
{
	Error(instruction, "unsupported architecture");
}

}
}
