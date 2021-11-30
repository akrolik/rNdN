#include "Backend/Codegen/Generators/Instructions/Logical/OrGenerator.h"

#include "Backend/Codegen/Generators/ArchitectureDispatch.h"

namespace Backend {
namespace Codegen {

void OrGenerator::Generate(const PTX::_OrInstruction *instruction)
{
	instruction->Dispatch(*this);
}

template<class T>
void OrGenerator::Visit(const PTX::OrInstruction<T> *instruction)
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
void OrGenerator::GenerateMaxwell(const PTX::OrInstruction<T> *instruction)
{
	this->GenerateLogicMaxwell(instruction,
		SASS::Maxwell::PSETPInstruction::BooleanOperator1::OR,
		SASS::Maxwell::LOPInstruction::BooleanOperator::OR
	);
}

template<class T>
void OrGenerator::GenerateVolta(const PTX::OrInstruction<T> *instruction)
{
	this->GenerateLogicVolta(instruction,
		[](std::uint8_t A, std::uint8_t B, std::uint8_t C) // Predicate function
		{
			return ((A | B) & C);
		},
		[](std::uint8_t A, std::uint8_t B, std::uint8_t C) // Integer function
		{
			return ((A | B) | C);
		}
	);
}

}
}
