#pragma once

#include "Test.h"

#include "PTX/Instructions/Shift/FunnelShiftInstruction.h"
#include "PTX/Instructions/Shift/ShiftLeftInstruction.h"
#include "PTX/Instructions/Shift/ShiftRightInstruction.h"

namespace Test {

class ShiftTest : public Test
{
public:
	void Execute()
	{
		PTX::RegisterSpace<PTX::Bit32Type> *b32 = new PTX::RegisterSpace<PTX::Bit32Type>("%b", 4);
		PTX::Register<PTX::Bit32Type> *regb32 = b32->GetVariable("%b", 0);

		PTX::RegisterSpace<PTX::UInt32Type> *u32 = new PTX::RegisterSpace<PTX::UInt32Type>("%u", 4);
		PTX::Register<PTX::UInt32Type> *regu32 = u32->GetVariable("%u", 0);

		PTX::ShiftLeftInstruction<PTX::Bit32Type> *test0 = new PTX::ShiftLeftInstruction<PTX::Bit32Type>(regb32, regb32, regu32);
		std::cout << test0->ToString() << std::endl;

		PTX::ShiftRightInstruction<PTX::Bit32Type> *test1 = new PTX::ShiftRightInstruction<PTX::Bit32Type>(regb32, regb32, regu32);
		std::cout << test1->ToString() << std::endl;

		PTX::FunnelShiftInstruction *test2 = new PTX::FunnelShiftInstruction(regb32, regb32, regb32, regu32, PTX::FunnelShiftInstruction::Direction::Left, PTX::FunnelShiftInstruction::Mode::Wrap);
		std::cout << test2->ToString() << std::endl;
	}
};

}
