#include "Backend/Codegen/Generators/InstructionGenerator.h"

#include "Backend/Codegen/Generators/Instructions/Arithmetic/AddGenerator.h"
#include "Backend/Codegen/Generators/Instructions/Arithmetic/MADGenerator.h"
#include "Backend/Codegen/Generators/Instructions/Arithmetic/MultiplyGenerator.h"
#include "Backend/Codegen/Generators/Instructions/Arithmetic/MultiplyWideGenerator.h"
#include "Backend/Codegen/Generators/Instructions/Arithmetic/RemainderGenerator.h"

#include "Backend/Codegen/Generators/Instructions/Comparison/SetPredicateGenerator.h"

#include "Backend/Codegen/Generators/Instructions/ControlFlow/BranchGenerator.h"
#include "Backend/Codegen/Generators/Instructions/ControlFlow/ReturnGenerator.h"

#include "Backend/Codegen/Generators/Instructions/Data/ConvertGenerator.h"
#include "Backend/Codegen/Generators/Instructions/Data/ConvertToAddressGenerator.h"
#include "Backend/Codegen/Generators/Instructions/Data/LoadGenerator.h"
#include "Backend/Codegen/Generators/Instructions/Data/LoadNCGenerator.h"
#include "Backend/Codegen/Generators/Instructions/Data/MoveSpecialGenerator.h"
#include "Backend/Codegen/Generators/Instructions/Data/PackGenerator.h"
#include "Backend/Codegen/Generators/Instructions/Data/StoreGenerator.h"
#include "Backend/Codegen/Generators/Instructions/Data/UnpackGenerator.h"

namespace Backend {
namespace Codegen {

// Arithmetic

void InstructionGenerator::Visit(const PTX::_AddInstruction *instruction)
{
	AddGenerator generator(this->m_builder);
	generator.Generate(instruction);
}

void InstructionGenerator::Visit(const PTX::_MADInstruction *instruction)
{
	MADGenerator generator(this->m_builder);
	generator.Generate(instruction);
}

void InstructionGenerator::Visit(const PTX::_MultiplyInstruction *instruction)
{
	MultiplyGenerator generator(this->m_builder);
	generator.Generate(instruction);
}

void InstructionGenerator::Visit(const PTX::_MultiplyWideInstruction *instruction)
{
	MultiplyWideGenerator generator(this->m_builder);
	generator.Generate(instruction);
}

void InstructionGenerator::Visit(const PTX::_RemainderInstruction *instruction)
{
	RemainderGenerator generator(this->m_builder);
	generator.Generate(instruction);
}

// Comparison

void InstructionGenerator::Visit(const PTX::_SetPredicateInstruction *instruction)
{
	SetPredicateGenerator generator(this->m_builder);
	generator.Generate(instruction);
}

// Control Flow

void InstructionGenerator::Visit(const PTX::BranchInstruction *instruction)
{
	BranchGenerator generator(this->m_builder);
	generator.Generate(instruction);
}

void InstructionGenerator::Visit(const PTX::ReturnInstruction *instruction)
{
	ReturnGenerator generator(this->m_builder);
	generator.Generate(instruction);
}

// Data

void InstructionGenerator::Visit(const PTX::_ConvertInstruction *instruction)
{
	ConvertGenerator generator(this->m_builder);
	generator.Generate(instruction);
}

void InstructionGenerator::Visit(const PTX::_ConvertToAddressInstruction *instruction)
{
	ConvertToAddressGenerator generator(this->m_builder);
	generator.Generate(instruction);
}

void InstructionGenerator::Visit(const PTX::_LoadInstruction *instruction)
{
	LoadGenerator generator(this->m_builder);
	generator.Generate(instruction);
}

void InstructionGenerator::Visit(const PTX::_LoadNCInstruction *instruction)
{
	LoadNCGenerator generator(this->m_builder);
	generator.Generate(instruction);
}

void InstructionGenerator::Visit(const PTX::_MoveSpecialInstruction *instruction)
{
	MoveSpecialGenerator generator(this->m_builder);
	generator.Generate(instruction);
}

void InstructionGenerator::Visit(const PTX::_PackInstruction *instruction)
{
	PackGenerator generator(this->m_builder);
	generator.Generate(instruction);
}

void InstructionGenerator::Visit(const PTX::_StoreInstruction *instruction)
{
	StoreGenerator generator(this->m_builder);
	generator.Generate(instruction);
}

void InstructionGenerator::Visit(const PTX::_UnpackInstruction *instruction)
{
	UnpackGenerator generator(this->m_builder);
	generator.Generate(instruction);
}

}
}
