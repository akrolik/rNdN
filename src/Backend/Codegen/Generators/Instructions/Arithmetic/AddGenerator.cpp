#include "Backend/Codegen/Generators/Instructions/Arithmetic/AddGenerator.h"

#include "Backend/Codegen/Generators/Operands/CompositeGenerator.h"
#include "Backend/Codegen/Generators/Operands/RegisterGenerator.h"

namespace Backend {
namespace Codegen {

void AddGenerator::Generate(const PTX::_AddInstruction *instruction)
{
	instruction->Dispatch(*this);
}

template<class T>
void AddGenerator::Visit(const PTX::AddInstruction<T> *instruction)
{
	// Setup operands

	RegisterGenerator registerGenerator(this->m_builder);
	CompositeGenerator compositeGenerator(this->m_builder);

	const auto [destination, destination_Hi] = registerGenerator.Generate(instruction->GetDestination());
	const auto [sourceA, sourceA_Hi] = registerGenerator.Generate(instruction->GetSourceA());
	const auto [sourceB, sourceB_Hi] = compositeGenerator.Generate(instruction->GetSourceB());

	// Generate instruction
	//  - Types:
	//  	- Int16, Int32, Int64
	//  	- UInt16, UInt32, UInt64
	//  	- Float16, Float16x2, Float32, Float64
	//  - Modifiers:
	//  	- Saturate: Int32, Float16, Float16x2, Float32
	//  	- Rounding: Float16, Float16x2, Float32, Float64
	//  	- FlushSubnormal: Float16, Float16x2, Float32
	//  	- Carry: Int32, Int64, UInt32, UInt64

	//TODO: Instruction Add<T> types/modifiers
	if constexpr(std::is_same<T, PTX::Int16Type>::value)
	{
		this->AddInstruction(new SASS::IADDInstruction(destination, sourceA, sourceB));
	}
	else if constexpr(std::is_same<T, PTX::UInt64Type>::value)
	{
		this->AddInstruction(new SASS::IADDInstruction(destination, sourceA, sourceB, SASS::IADDInstruction::Flags::CC));
		this->AddInstruction(new SASS::IADDInstruction(destination_Hi, sourceA_Hi, sourceB_Hi, SASS::IADDInstruction::Flags::X));
	}
}

}
}
