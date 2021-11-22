#include "Backend/Codegen/Generators/Instructions/Shift/ShiftRightGenerator.h"

#include "Backend/Codegen/Generators/Operands/CompositeGenerator.h"
#include "Backend/Codegen/Generators/Operands/RegisterGenerator.h"

#include "Backend/Codegen/Generators/ArchitectureDispatch.h"

namespace Backend {
namespace Codegen {

void ShiftRightGenerator::Generate(const PTX::_ShiftRightInstruction *instruction)
{
	instruction->Dispatch(*this);
}

template<class T>
void ShiftRightGenerator::Visit(const PTX::ShiftRightInstruction<T> *instruction)
{
	// Instruction predicate

	this->SetPredicatedInstruction(instruction);

	// Types:
	//   - Bit16, Bit32, Bit64    
	//   - UInt16, UInt32, UInt64    
	//   - Int16, Int32, Int64    
	// Modifiers: --

	ArchitectureDispatch::Dispatch(*this, instruction);
}

template<class T>
void ShiftRightGenerator::GenerateMaxwell(const PTX::ShiftRightInstruction<T> *instruction)
{
	if constexpr(T::TypeBits == PTX::Bits::Bits32)
	{
		// Generate operands

		RegisterGenerator registerGenerator(this->m_builder);
		auto destination = registerGenerator.Generate(instruction->GetDestination());
		auto sourceA = registerGenerator.Generate(instruction->GetSourceA());

		CompositeGenerator compositeGenerator(this->m_builder);
		auto sourceB = compositeGenerator.Generate(instruction->GetSourceB());

		// Flags

		auto flags = SASS::Maxwell::SHRInstruction::Flags::None;
		if constexpr(PTX::is_unsigned_int_type<T>::value || PTX::is_bit_type<T>::value)
		{
			flags |= SASS::Maxwell::SHRInstruction::Flags::U32;
		}

		// Generate instruction

		this->AddInstruction(new SASS::Maxwell::SHRInstruction(destination, sourceA, sourceB, flags));
	}
	else
	{
		Error(instruction, "unsupported type");
	}
}

template<class T>
void ShiftRightGenerator::GenerateVolta(const PTX::ShiftRightInstruction<T> *instruction)
{
	Error(instruction, "unsupported architecture");
}

}
}
