#include "Backend/Codegen/Generators/Instructions/Data/UnpackGenerator.h"

#include "Backend/Codegen/Generators/Operands/RegisterGenerator.h"

#include "Backend/Codegen/Generators/ArchitectureDispatch.h"

namespace Backend {
namespace Codegen {

void UnpackGenerator::Generate(const PTX::_UnpackInstruction *instruction)
{
	instruction->Dispatch(*this);
}

template<class T, PTX::VectorSize V>
void UnpackGenerator::Visit(const PTX::UnpackInstruction<T, V> *instruction)
{
	// Instruction predicate

	this->SetPredicatedInstruction(instruction);

	// Types:
	//   - Bit16, Bit32, Bit64
	// Vector
	//   - Vector2
	//   - Vector4

	ArchitectureDispatch::DispatchInstruction<
		SASS::Maxwell::MOVInstruction, SASS::Volta::MOVInstruction
	>(*this, instruction);
}

template<class MOVInstruction, class T, PTX::VectorSize V>
void UnpackGenerator::GenerateInstruction(const PTX::UnpackInstruction<T, V> *instruction)
{
	// Generate source register

	RegisterGenerator registerGenerator(this->m_builder);
	auto [source_Lo, source_Hi] = registerGenerator.GeneratePair(instruction->GetSource());

	// Destination decomposition, split below

	auto destinations = instruction->GetDestination()->GetRegisters();

	// Generate instruction

	if constexpr(V == PTX::VectorSize::Vector2)
	{
		// Generate destination registers

		auto destinationA = registerGenerator.Generate(destinations.at(0));
		auto destinationB = registerGenerator.Generate(destinations.at(1));

		// Temporary necessary for register reuse

		auto temp = this->m_builder.AllocateTemporaryRegister();

		if constexpr(std::is_same<T, PTX::Bit16Type>::value || std::is_same<T, PTX::Bit32Type>::value)
		{
			// Unpack by shifting (for high bits) and masking (for low bits)

			auto shift = PTX::BitSize<T::TypeBits>::NumBits / 2;
			auto mask = (1 << shift) - 1;

			ArchitectureDispatch::DispatchInline(this->m_builder,
			[&]() // Maxwell instruction set
			{
				this->AddInstruction(new SASS::Maxwell::SHRInstruction(
					temp, source_Lo, new SASS::I32Immediate(shift), SASS::Maxwell::SHRInstruction::Flags::U32
				));
				this->AddInstruction(new SASS::Maxwell::LOPInstruction(
					destinationA, source_Lo, new SASS::I32Immediate(mask), SASS::Maxwell::LOPInstruction::BooleanOperator::AND
				));
			},
			[&]() // Volta instruction set
			{
				auto logicOperation = SASS::Volta::BinaryUtils::LogicOperation(
					[](std::uint8_t A, std::uint8_t B, std::uint8_t C)
					{
						return ((A & B) | C);
					}
				);

				this->AddInstruction(new SASS::Volta::SHFInstruction(
					temp, SASS::RZ, new SASS::I32Immediate(shift), source_Lo, 
					SASS::Volta::SHFInstruction::Direction::R,
					SASS::Volta::SHFInstruction::Type::U32,
					SASS::Volta::SHFInstruction::Flags::HI
				));
				this->AddInstruction(new SASS::Volta::LOP3Instruction(
					destinationA, source_Lo, new SASS::I32Immediate(mask), SASS::RZ, new SASS::I8Immediate(logicOperation), SASS::PT
				));
			});

			this->AddInstruction(new MOVInstruction(destinationB, temp));
		}
		else if constexpr(std::is_same<T, PTX::Bit64Type>::value)
		{
			if (destinationA->GetValue() != source_Hi->GetValue())
			{
				this->AddInstruction(new MOVInstruction(destinationA, source_Lo));
				this->AddInstruction(new MOVInstruction(destinationB, source_Hi));
			}
			else
			{
				this->AddInstruction(new MOVInstruction(temp, source_Lo));
				this->AddInstruction(new MOVInstruction(destinationB, source_Hi));
				this->AddInstruction(new MOVInstruction(destinationA, temp));
			}
		}
	}
	else if constexpr(V == PTX::VectorSize::Vector4)
	{
		Error(instruction, "unsupported vector size");
	}
}

}
}
