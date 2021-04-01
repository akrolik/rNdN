#include "Backend/Codegen/Generators/Instructions/Shift/ShiftLeftGenerator.h"

#include "Backend/Codegen/Generators/Operands/CompositeGenerator.h"
#include "Backend/Codegen/Generators/Operands/RegisterGenerator.h"

namespace Backend {
namespace Codegen {

void ShiftLeftGenerator::Generate(const PTX::_ShiftLeftInstruction *instruction)
{
	instruction->Dispatch(*this);
}

template<class T>
void ShiftLeftGenerator::Visit(const PTX::ShiftLeftInstruction<T> *instruction)
{
	// Types:
	//   - Bit16, Bit32, Bit64    
	// Modifiers: --

	if constexpr(std::is_same<T, PTX::Bit32Type>::value)
	{
		// Generate operands

		RegisterGenerator registerGenerator(this->m_builder);
		auto [destination, destination_Hi] = registerGenerator.Generate(instruction->GetDestination());
		auto [sourceA, sourceA_Hi] = registerGenerator.Generate(instruction->GetSourceA());

		CompositeGenerator compositeGenerator(this->m_builder);
		auto [sourceB, sourceB_Hi] = compositeGenerator.Generate(instruction->GetSourceB());

		// Flags

		auto flags = SASS::SHLInstruction::Flags::None;

		// Generate instruction

		this->AddInstruction(new SASS::SHLInstruction(destination, sourceA, sourceB, flags));
	}
	else
	{
		Error(instruction, "unsupported type");
	}
}

}
}
