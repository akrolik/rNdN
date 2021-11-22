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

	ArchitectureDispatch::Dispatch(*this, instruction);
}

template<class T, PTX::VectorSize V>
void UnpackGenerator::GenerateMaxwell(const PTX::UnpackInstruction<T, V> *instruction)
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

		if constexpr(std::is_same<T, PTX::Bit16Type>::value)
		{
			this->AddInstruction(new SASS::Maxwell::SHRInstruction(
				temp, source_Lo, new SASS::I32Immediate(0x8), SASS::Maxwell::SHRInstruction::Flags::U32
			));
			this->AddInstruction(new SASS::Maxwell::LOPInstruction(
				destinationA, source_Lo, new SASS::I32Immediate(0xff), SASS::Maxwell::LOPInstruction::BooleanOperator::AND
			));
			this->AddInstruction(new SASS::Maxwell::MOVInstruction(destinationB, temp));
		}
		else if constexpr(std::is_same<T, PTX::Bit32Type>::value)
		{
			this->AddInstruction(new SASS::Maxwell::SHRInstruction(
				temp, source_Lo, new SASS::I32Immediate(0x10), SASS::Maxwell::SHRInstruction::Flags::U32
			));
			this->AddInstruction(new SASS::Maxwell::LOPInstruction(
				destinationA, source_Lo, new SASS::I32Immediate(0xffff), SASS::Maxwell::LOPInstruction::BooleanOperator::AND
			));
			this->AddInstruction(new SASS::Maxwell::MOVInstruction(destinationB, temp));
		}
		else if constexpr(std::is_same<T, PTX::Bit64Type>::value)
		{
			if (destinationA->GetValue() != source_Hi->GetValue())
			{
				this->AddInstruction(new SASS::Maxwell::MOVInstruction(destinationA, source_Lo));
				this->AddInstruction(new SASS::Maxwell::MOVInstruction(destinationB, source_Hi));
			}
			else
			{
				this->AddInstruction(new SASS::Maxwell::MOVInstruction(temp, source_Lo));
				this->AddInstruction(new SASS::Maxwell::MOVInstruction(destinationB, source_Hi));
				this->AddInstruction(new SASS::Maxwell::MOVInstruction(destinationA, temp));
			}
		}
	}
	else if constexpr(V == PTX::VectorSize::Vector4)
	{
		Error(instruction, "unsupported vector size");
	}
}

template<class T, PTX::VectorSize V>
void UnpackGenerator::GenerateVolta(const PTX::UnpackInstruction<T, V> *instruction)
{
	Error(instruction, "unsupported architecture");
}

}
}
