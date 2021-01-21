#include "Backend/Codegen/Generators/InstructionGenerator.h"

#include "Backend/Codegen/Generators/Instructions/ControlFlow/BranchGenerator.h"
#include "Backend/Codegen/Generators/Instructions/ControlFlow/ReturnGenerator.h"

#include "Backend/Codegen/Generators/Instructions/Data/MoveSpecialGenerator.h"

namespace Backend {
namespace Codegen {

// Arithmetic

void InstructionGenerator::Visit(const PTX::_AddInstruction *instruction)
{
	// add.u64 $ud4, %ud_$data$t1, $ud5;
	// add.s16 $rs0, $rs1, 0x1;
}

void InstructionGenerator::Visit(const PTX::_MADInstruction *instruction)
{
	// mad.lo.u32 $u3, $u1, $u2, $u0;
}

void InstructionGenerator::Visit(const PTX::_MultiplyWideInstruction *instruction)
{
	// mul.wide.u32 $ud5, $u3, 0x1;
}

// Comparison

void InstructionGenerator::Visit(const PTX::_SetPredicateInstruction *instruction)
{
	// setp.ge.u32 $p0, $u3, %u_$size$t1;
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
	// cvt.s16.s8 $rs1, %rc_t1$vector;
	// cvt.s8.s16 %rc_t2, $rs0;
}

void InstructionGenerator::Visit(const PTX::_ConvertAddressInstruction *instruction)
{
	// cvta.to.global.u64 %ud_$data$t1, $ud0;
}

void InstructionGenerator::Visit(const PTX::_LoadInstruction *instruction)
{
	// ld.param.u32 %u_$geometry$size, [$geometry$size];
	// ld.param.u64 $ud0, [t1];
	// ld.global.s8 %rc_t1$vector, [$ud4];
}

void InstructionGenerator::Visit(const PTX::_LoadNCInstruction *instruction)
{
	// ld.global.nc.u32 %u_$size$t1, [%ud_$data$$size$t1];
}

void InstructionGenerator::Visit(const PTX::_MoveSpecialInstruction *instruction)
{
	MoveSpecialGenerator generator(this->m_builder);
	generator.Generate(instruction);
}

void InstructionGenerator::Visit(const PTX::_PackInstruction *instruction)
{
	// mov.b16 $bs, {0x0, 0};
}

void InstructionGenerator::Visit(const PTX::_StoreInstruction *instruction)
{
	// st.global.s8 [$ud6], %rc_t2;
}

void InstructionGenerator::Visit(const PTX::_UnpackInstruction *instruction)
{
	// mov.b16 {%rc_t1$vector, _}, $bs;
}

}
}
