#pragma once

#include "Test.h"

#include "PTX/Declarations/VariableDeclaration.h"

#include "PTX/Instructions/Shift/FunnelShiftInstruction.h"
#include "PTX/Instructions/Shift/ShiftLeftInstruction.h"
#include "PTX/Instructions/Shift/ShiftRightInstruction.h"

namespace Test {

class ShiftTest : public Test
{
public:
	void Execute()
	{
		PTX::RegisterDeclaration<PTX::Bit32Type> *b32 = new PTX::RegisterDeclaration<PTX::Bit32Type>("%b", 4);
		PTX::RegisterDeclaration<PTX::UInt32Type> *u32 = new PTX::RegisterDeclaration<PTX::UInt32Type>("%u", 4);

		const PTX::Register<PTX::Bit32Type> *reg_b32 = b32->GetVariable("%b", 0);
		const PTX::Register<PTX::UInt32Type> *reg_u32 = u32->GetVariable("%u", 0);

		PTX::ShiftLeftInstruction<PTX::Bit32Type> *test1 = new PTX::ShiftLeftInstruction<PTX::Bit32Type>(reg_b32, reg_b32, reg_u32);
		std::cout << test1->ToString(0) << std::endl;

		PTX::ShiftRightInstruction<PTX::Bit32Type> *test2 = new PTX::ShiftRightInstruction<PTX::Bit32Type>(reg_b32, reg_b32, reg_u32);
		std::cout << test2->ToString(0) << std::endl;

		PTX::FunnelShiftInstruction *test3 = new PTX::FunnelShiftInstruction(reg_b32, reg_b32, reg_b32, reg_u32, PTX::FunnelShiftInstruction::Direction::Left, PTX::FunnelShiftInstruction::Mode::Wrap);
		std::cout << test3->ToString(0) << std::endl;
	}
};

}
