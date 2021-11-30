#include "Backend/Codegen/Generators/Instructions/Logical/AndGenerator.h"

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
	this->GenerateLogicMaxwell(instruction,
		SASS::Maxwell::PSETPInstruction::BooleanOperator1::AND,
		SASS::Maxwell::LOPInstruction::BooleanOperator::AND
	);
}

template<class T>
void AndGenerator::GenerateVolta(const PTX::AndInstruction<T> *instruction)
{
	this->GenerateLogicVolta(instruction,
		[](std::uint8_t A, std::uint8_t B, std::uint8_t C) // Predicate function
		{
			return ((A & B) & C);
		},
		[](std::uint8_t A, std::uint8_t B, std::uint8_t C) // Integer function
		{
			return ((A & B) | C);
		}
	);
}

}
}
