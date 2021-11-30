#include "Backend/Codegen/Generators/Instructions/Data/MoveGenerator.h"

#include "Backend/Codegen/Generators/Operands/CompositeGenerator.h"
#include "Backend/Codegen/Generators/Operands/RegisterGenerator.h"
#include "Backend/Codegen/Generators/Operands/PredicateGenerator.h"

#include "Backend/Codegen/Generators/ArchitectureDispatch.h"

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

		ArchitectureDispatch::DispatchInline(this->m_builder, 
		[&]() // Maxwell instruction set
		{
			// Flags

			auto flags = SASS::Maxwell::PSETPInstruction::Flags::None;
			if (source_Not)
			{
				flags |= SASS::Maxwell::PSETPInstruction::Flags::NOT_A;
			}

			// Generate instruction

			this->AddInstruction(new SASS::Maxwell::PSETPInstruction(
				destination, SASS::PT, source, SASS::PT, SASS::PT,
				SASS::Maxwell::PSETPInstruction::BooleanOperator1::AND,
				SASS::Maxwell::PSETPInstruction::BooleanOperator2::AND,
				flags
			));
		},
		[&]() // Volta instruction set
		{
			// Flags

			auto flags = SASS::Volta::PLOP3Instruction::Flags::None;
			if (source_Not)
			{
				flags |= SASS::Volta::PLOP3Instruction::Flags::NOT_A;
			}

			// Generate instruction

			auto logicOperation = SASS::Volta::BinaryUtils::LogicOperation(
				[](std::uint8_t A, std::uint8_t B, std::uint8_t C)
				{
					return ((A & B) & C);
				}
			);

			this->AddInstruction(new SASS::Volta::PLOP3Instruction(
				destination, SASS::PT, source, SASS::PT, SASS::PT,
				new SASS::I8Immediate(logicOperation), new SASS::I8Immediate(0x0), flags
			));
		});
	}
	else
	{
		ArchitectureDispatch::DispatchInstruction<
			SASS::Maxwell::MOVInstruction, SASS::Volta::MOVInstruction
		>(*this, instruction);
	}
}

template<class MOVInstruction, class T>
void MoveGenerator::GenerateInstruction(const PTX::MoveInstruction<T> *instruction)
{
	// Generate operands

	RegisterGenerator registerGenerator(this->m_builder);
	auto [destination_Lo, destination_Hi] = registerGenerator.GeneratePair(instruction->GetDestination());

	CompositeGenerator compositeGenerator(this->m_builder);
	if constexpr(std::is_same<MOVInstruction, SASS::Volta::MOVInstruction>::value)
	{
		compositeGenerator.SetImmediateSize(32);
	}
	auto [source_Lo, source_Hi] = compositeGenerator.GeneratePair(instruction->GetSource());

	// Generate instruction (no overlap unless equal)

	this->AddInstruction(new MOVInstruction(destination_Lo, source_Lo));
	if constexpr(T::TypeBits == PTX::Bits::Bits64)
	{
		this->AddInstruction(new MOVInstruction(destination_Hi, source_Hi));
	}
}

}
}
