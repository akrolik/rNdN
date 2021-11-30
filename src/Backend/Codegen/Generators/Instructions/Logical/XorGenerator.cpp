#include "Backend/Codegen/Generators/Instructions/Logical/XorGenerator.h"

#include "Backend/Codegen/Generators/ArchitectureDispatch.h"

namespace Backend {
namespace Codegen {

void XorGenerator::Generate(const PTX::_XorInstruction *instruction)
{
	instruction->Dispatch(*this);
}

template<class T>
void XorGenerator::Visit(const PTX::XorInstruction<T> *instruction)
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
void XorGenerator::GenerateMaxwell(const PTX::XorInstruction<T> *instruction)
{
	this->GenerateLogicMaxwell(instruction,
		SASS::Maxwell::PSETPInstruction::BooleanOperator1::XOR,
		SASS::Maxwell::LOPInstruction::BooleanOperator::XOR
	);
}

template<class T>
void XorGenerator::GenerateVolta(const PTX::XorInstruction<T> *instruction)
{
	this->GenerateLogicVolta(instruction,
		[](std::uint8_t A, std::uint8_t B, std::uint8_t C) // Predicate function
		{
			return ((A ^ B) & C);
		},
		[](std::uint8_t A, std::uint8_t B, std::uint8_t C) // Integer function
		{
			return ((A ^ B) | C);
		}
	);
}

}
}
