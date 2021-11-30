#include "Backend/Codegen/Generators/Instructions/Shift/ShiftLeftGenerator.h"

#include "Backend/Codegen/Generators/Operands/CompositeGenerator.h"
#include "Backend/Codegen/Generators/Operands/RegisterGenerator.h"

#include "Backend/Codegen/Generators/ArchitectureDispatch.h"

namespace Backend {
namespace Codegen {

void ShiftLeftGenerator::Generate(const PTX::_ShiftLeftInstruction *instruction)
{
	instruction->Dispatch(*this);
}

template<class T>
void ShiftLeftGenerator::Visit(const PTX::ShiftLeftInstruction<T> *instruction)
{
	// Instruction predicate

	this->SetPredicatedInstruction(instruction);

	// Types:
	//   - Bit16, Bit32, Bit64    
	// Modifiers: --

	ArchitectureDispatch::Dispatch(*this, instruction);
}

template<class T>
void ShiftLeftGenerator::GenerateMaxwell(const PTX::ShiftLeftInstruction<T> *instruction)
{
	if constexpr(std::is_same<T, PTX::Bit32Type>::value)
	{
		// Generate operands

		RegisterGenerator registerGenerator(this->m_builder);
		auto destination = registerGenerator.Generate(instruction->GetDestination());
		auto sourceA = registerGenerator.Generate(instruction->GetSourceA());

		CompositeGenerator compositeGenerator(this->m_builder);
		auto sourceB = compositeGenerator.Generate(instruction->GetSourceB());

		// Flags

		auto flags = SASS::Maxwell::SHLInstruction::Flags::None;

		// Generate instruction

		this->AddInstruction(new SASS::Maxwell::SHLInstruction(destination, sourceA, sourceB, flags));
	}
	else
	{
		Error(instruction, "unsupported type");
	}
}

template<class T>
void ShiftLeftGenerator::GenerateVolta(const PTX::ShiftLeftInstruction<T> *instruction)
{
	if constexpr(std::is_same<T, PTX::Bit32Type>::value)
	{
		// Generate operands

		RegisterGenerator registerGenerator(this->m_builder);
		auto destination = registerGenerator.Generate(instruction->GetDestination());
		auto sourceA = registerGenerator.Generate(instruction->GetSourceA());

		CompositeGenerator compositeGenerator(this->m_builder);
		compositeGenerator.SetImmediateSize(32);
		auto sourceB = compositeGenerator.Generate(instruction->GetSourceB());

		// Flags

		auto direction = SASS::Volta::SHFInstruction::Direction::L;
		auto type = SASS::Volta::SHFInstruction::Type::U32;

		// Generate instruction

		this->AddInstruction(new SASS::Volta::SHFInstruction(destination, sourceA, sourceB, SASS::RZ, direction, type));
	}
	else
	{
		Error(instruction, "unsupported type");
	}
}

}
}
