#pragma once

#include "Test.h"

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
		PTX::RegisterSpace<PTX::Bit32Type> *b32 = new PTX::RegisterSpace<PTX::Bit32Type>("%b", 4);
		PTX::Register<PTX::Bit32Type> *regb32 = b32->GetVariable("%b", 0);

		PTX::AndInstruction<PTX::Bit32Type> *test0 = new PTX::AndInstruction<PTX::Bit32Type>(regb32, regb32, regb32);
		std::cout << test0->ToString() << std::endl;

		PTX::Logical3OpInstruction *test1 = new PTX::Logical3OpInstruction(regb32, regb32, regb32, regb32, 0x80);
		std::cout << test1->ToString() << std::endl;

	}
};

}
