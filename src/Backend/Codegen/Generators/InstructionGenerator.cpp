#include "Backend/Codegen/Generators/InstructionGenerator.h"

#include "Backend/Codegen/Generators/Instructions/Arithmetic/AddGenerator.h"
#include "Backend/Codegen/Generators/Instructions/Arithmetic/CountLeadingZerosGenerator.h"
#include "Backend/Codegen/Generators/Instructions/Arithmetic/DivideGenerator.h"
#include "Backend/Codegen/Generators/Instructions/Arithmetic/MADGenerator.h"
#include "Backend/Codegen/Generators/Instructions/Arithmetic/MultiplyGenerator.h"
#include "Backend/Codegen/Generators/Instructions/Arithmetic/MultiplyWideGenerator.h"
#include "Backend/Codegen/Generators/Instructions/Arithmetic/RemainderGenerator.h"
#include "Backend/Codegen/Generators/Instructions/Arithmetic/SubtractGenerator.h"

#include "Backend/Codegen/Generators/Instructions/Comparison/SelectGenerator.h"
#include "Backend/Codegen/Generators/Instructions/Comparison/SetPredicateGenerator.h"

#include "Backend/Codegen/Generators/Instructions/ControlFlow/BranchGenerator.h"
#include "Backend/Codegen/Generators/Instructions/ControlFlow/ReturnGenerator.h"

#include "Backend/Codegen/Generators/Instructions/Data/ConvertGenerator.h"
#include "Backend/Codegen/Generators/Instructions/Data/ConvertToAddressGenerator.h"
#include "Backend/Codegen/Generators/Instructions/Data/LoadGenerator.h"
#include "Backend/Codegen/Generators/Instructions/Data/LoadNCGenerator.h"
#include "Backend/Codegen/Generators/Instructions/Data/MoveGenerator.h"
#include "Backend/Codegen/Generators/Instructions/Data/MoveAddressGenerator.h"
#include "Backend/Codegen/Generators/Instructions/Data/MoveSpecialGenerator.h"
#include "Backend/Codegen/Generators/Instructions/Data/PackGenerator.h"
#include "Backend/Codegen/Generators/Instructions/Data/ShuffleGenerator.h"
#include "Backend/Codegen/Generators/Instructions/Data/StoreGenerator.h"
#include "Backend/Codegen/Generators/Instructions/Data/UnpackGenerator.h"

#include "Backend/Codegen/Generators/Instructions/Logical/AndGenerator.h"
#include "Backend/Codegen/Generators/Instructions/Logical/NotGenerator.h"
#include "Backend/Codegen/Generators/Instructions/Logical/OrGenerator.h"
#include "Backend/Codegen/Generators/Instructions/Logical/XorGenerator.h"

#include "Backend/Codegen/Generators/Instructions/Shift/ShiftLeftGenerator.h"
#include "Backend/Codegen/Generators/Instructions/Shift/ShiftRightGenerator.h"

#include "Backend/Codegen/Generators/Instructions/Synchronization/AtomicGenerator.h"
#include "Backend/Codegen/Generators/Instructions/Synchronization/BarrierGenerator.h"
#include "Backend/Codegen/Generators/Instructions/Synchronization/ReductionGenerator.h"

namespace Backend {
namespace Codegen {

// Arithmetic

void InstructionGenerator::Visit(const PTX::_AbsoluteInstruction *instruction)
{
	Error("AbsoluteInstruction");
}

void InstructionGenerator::Visit(const PTX::_AddInstruction *instruction)
{
	AddGenerator generator(this->m_builder);
	generator.Generate(instruction);
}

void InstructionGenerator::Visit(const PTX::_CountLeadingZerosInstruction *instruction)
{
	CountLeadingZerosGenerator generator(this->m_builder);
	generator.Generate(instruction);
}

void InstructionGenerator::Visit(const PTX::_DivideInstruction *instruction)
{
	DivideGenerator generator(this->m_builder);
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

void InstructionGenerator::Visit(const PTX::_NegateInstruction *instruction)
{
	Error("NegateInstruction");
}

void InstructionGenerator::Visit(const PTX::_ReciprocalInstruction *instruction)
{
	Error("ReciprocalInstruction");
}

void InstructionGenerator::Visit(const PTX::_RemainderInstruction *instruction)
{
	RemainderGenerator generator(this->m_builder);
	generator.Generate(instruction);
}

void InstructionGenerator::Visit(const PTX::_SubtractInstruction *instruction)
{
	SubtractGenerator generator(this->m_builder);
	generator.Generate(instruction);
}

// Comparison

void InstructionGenerator::Visit(const PTX::_SelectInstruction *instruction)
{
	SelectGenerator generator(this->m_builder);
	generator.Generate(instruction);
}

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

void InstructionGenerator::Visit(const PTX::_CallInstructionBase *instruction)
{
	Error("CallInstruction");
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

void InstructionGenerator::Visit(const PTX::_MoveInstruction *instruction)
{
	MoveGenerator generator(this->m_builder);
	generator.Generate(instruction);
}

void InstructionGenerator::Visit(const PTX::_MoveAddressInstruction *instruction)
{
	MoveAddressGenerator generator(this->m_builder);
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

void InstructionGenerator::Visit(const PTX::_ShuffleInstruction *instruction)
{
	ShuffleGenerator generator(this->m_builder);
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

void InstructionGenerator::Visit(const PTX::_AndInstruction *instruction)
{
	AndGenerator generator(this->m_builder);
	generator.Generate(instruction);
}

void InstructionGenerator::Visit(const PTX::_NotInstruction *instruction)
{
	NotGenerator generator(this->m_builder);
	generator.Generate(instruction);
}

void InstructionGenerator::Visit(const PTX::_OrInstruction *instruction)
{
	OrGenerator generator(this->m_builder);
	generator.Generate(instruction);
}

void InstructionGenerator::Visit(const PTX::_XorInstruction *instruction)
{
	XorGenerator generator(this->m_builder);
	generator.Generate(instruction);
}

void InstructionGenerator::Visit(const PTX::_ShiftLeftInstruction *instruction)
{
	ShiftLeftGenerator generator(this->m_builder);
	generator.Generate(instruction);
}

void InstructionGenerator::Visit(const PTX::_ShiftRightInstruction *instruction)
{
	ShiftRightGenerator generator(this->m_builder);
	generator.Generate(instruction);
}

void InstructionGenerator::Visit(const PTX::_AtomicInstruction *instruction)
{
	AtomicGenerator generator(this->m_builder);
	generator.Generate(instruction);
}

void InstructionGenerator::Visit(const PTX::BarrierInstruction *instruction)
{
	BarrierGenerator generator(this->m_builder);
	generator.Generate(instruction);
}

void InstructionGenerator::Visit(const PTX::_ReductionInstruction *instruction)
{
	ReductionGenerator generator(this->m_builder);
	generator.Generate(instruction);
}

}
}
