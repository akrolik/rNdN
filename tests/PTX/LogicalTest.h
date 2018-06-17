#pragma once

#include "Test.h"

#include "PTX/Declarations/VariableDeclaration.h"

#include "PTX/Instructions/Logical/AndInstruction.h"
#include "PTX/Instructions/Logical/OrInstruction.h"
#include "PTX/Instructions/Logical/XorInstruction.h"
#include "PTX/Instructions/Logical/NotInstruction.h"
#include "PTX/Instructions/Logical/CNotInstruction.h"
#include "PTX/Instructions/Logical/Logical3OpInstruction.h"

namespace Test {

class LogicalTest : public Test
{
public:
	void Execute()
	{
		PTX::RegisterDeclaration<PTX::Bit32Type> *b32 = new PTX::RegisterDeclaration<PTX::Bit32Type>("%b", 4);
		const PTX::Register<PTX::Bit32Type> *reg_b32 = b32->GetVariable("%b", 0);

		PTX::AndInstruction<PTX::Bit32Type> *test1 = new PTX::AndInstruction<PTX::Bit32Type>(reg_b32, reg_b32, reg_b32);
		std::cout << test1->ToString() << std::endl;

		PTX::Logical3OpInstruction *test2 = new PTX::Logical3OpInstruction(reg_b32, reg_b32, reg_b32, reg_b32, 0x80);
		std::cout << test2->ToString() << std::endl;
	}
};

}
