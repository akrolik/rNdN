#include "Backend/Codegen/Generators/Instructions/Logical/NotGenerator.h"

#include "Backend/Codegen/Generators/Operands/CompositeGenerator.h"
#include "Backend/Codegen/Generators/Operands/PredicateGenerator.h"
#include "Backend/Codegen/Generators/Operands/RegisterGenerator.h"

#include "Backend/Codegen/Generators/ArchitectureDispatch.h"

namespace Backend {
namespace Codegen {

void NotGenerator::Generate(const PTX::_NotInstruction *instruction)
{
	instruction->Dispatch(*this);
}

template<class T>
void NotGenerator::Visit(const PTX::NotInstruction<T> *instruction)
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
void NotGenerator::GenerateMaxwell(const PTX::NotInstruction<T> *instruction)
{
	if constexpr(std::is_same<T, PTX::PredicateType>::value)
	{
		// Generate operands

		PredicateGenerator predicateGenerator(this->m_builder);
		auto destination = predicateGenerator.Generate(instruction->GetDestination()).first;
		auto [source, source_Not] = predicateGenerator.Generate(instruction->GetSource());

		// Flags

		auto flags = SASS::Maxwell::PSETPInstruction::Flags::NOT_A;
		if (source_Not)
		{
			flags = SASS::Maxwell::PSETPInstruction::Flags::None;
		}

		// Generate instruction

		this->AddInstruction(new SASS::Maxwell::PSETPInstruction(
			destination, SASS::PT, source, SASS::PT, SASS::PT,
			SASS::Maxwell::PSETPInstruction::BooleanOperator1::AND,
			SASS::Maxwell::PSETPInstruction::BooleanOperator2::AND,
			flags
		));
	}
	else
	{
		RegisterGenerator registerGenerator(this->m_builder);
		auto [destination_Lo, destination_Hi] = registerGenerator.GeneratePair(instruction->GetDestination());

		CompositeGenerator compositeGenerator(this->m_builder);
		auto [source_Lo, source_Hi] = compositeGenerator.GeneratePair(instruction->GetSource());

		this->AddInstruction(new SASS::Maxwell::LOPInstruction(
			destination_Lo, SASS::RZ, source_Lo,
			SASS::Maxwell::LOPInstruction::BooleanOperator::PASS_B,
			SASS::Maxwell::LOPInstruction::Flags::INV
		));

		if constexpr(T::TypeBits == PTX::Bits::Bits64)
		{
			this->AddInstruction(new SASS::Maxwell::LOPInstruction(
				destination_Hi, SASS::RZ, source_Hi,
				SASS::Maxwell::LOPInstruction::BooleanOperator::PASS_B,
				SASS::Maxwell::LOPInstruction::Flags::INV
			));
		}
	}
}

template<class T>
void NotGenerator::GenerateVolta(const PTX::NotInstruction<T> *instruction)
{
	if constexpr(std::is_same<T, PTX::PredicateType>::value)
	{
		// Generate operands

		PredicateGenerator predicateGenerator(this->m_builder);
		auto destination = predicateGenerator.Generate(instruction->GetDestination()).first;
		auto [source, source_Not] = predicateGenerator.Generate(instruction->GetSource());

		// Flags

		auto flags = SASS::Volta::PLOP3Instruction::Flags::NOT_A;
		if (source_Not)
		{
			flags = SASS::Volta::PLOP3Instruction::Flags::None;
		}

		// Generate instruction

		auto logicOperation = SASS::Volta::BinaryUtils::LogicOperation([](std::uint8_t A, std::uint8_t B, std::uint8_t C) // Predicate function
		{
			return ((~A) & B & C);
		});

		this->AddInstruction(new SASS::Volta::PLOP3Instruction(
			destination, SASS::PT, source, SASS::PT, SASS::PT,
			new SASS::I8Immediate(logicOperation), new SASS::I8Immediate(0x0), flags
		));
	}
	else
	{
		// Generate operands

		RegisterGenerator registerGenerator(this->m_builder);
		auto [destination_Lo, destination_Hi] = registerGenerator.GeneratePair(instruction->GetDestination());

		CompositeGenerator compositeGenerator(this->m_builder);
		compositeGenerator.SetImmediateSize(32);
		auto [source_Lo, source_Hi] = compositeGenerator.GeneratePair(instruction->GetSource());

		// Generate instruction

		auto logicOperation = SASS::Volta::BinaryUtils::LogicOperation([](std::uint8_t A, std::uint8_t B, std::uint8_t C) // Predicate function
		{
			return (A | (~B) | C);
		});

		this->AddInstruction(new SASS::Volta::LOP3Instruction(
			destination_Lo, SASS::RZ, source_Lo, SASS::RZ, new SASS::I8Immediate(logicOperation), SASS::PT
		));

		if constexpr(T::TypeBits == PTX::Bits::Bits64)
		{
			this->AddInstruction(new SASS::Volta::LOP3Instruction(
				destination_Hi, SASS::RZ, source_Hi, SASS::RZ, new SASS::I8Immediate(logicOperation), SASS::PT
			));
		}
	}
}

}
}
