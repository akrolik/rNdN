#pragma once

#include "Test.h"

#include "PTX/Instructions/Comparison/SetInstruction.h"
#include "PTX/Instructions/Comparison/SetPredicateInstruction.h"
#include "PTX/Instructions/Comparison/SelectInstruction.h"
#include "PTX/Instructions/Comparison/SignSelectInstruction.h"

namespace Test {

class ComparisonTest : public Test
{
public:
	void Execute()
	{
		// PTX::RegisterSpace<PTX::Bit32Type> *b32 = new PTX::RegisterSpace<PTX::Bit32Type>("%b", 4);
		// PTX::Register<PTX::Bit32Type> *regb32 = b32->GetVariable("%b", 0);

		// PTX::AndInstruction<PTX::Bit32Type> *test0 = new PTX::AndInstruction<PTX::Bit32Type>(regb32, regb32, regb32);
		// std::cout << test0->ToString() << std::endl;

		// PTX::Logical3OpInstruction *test1 = new PTX::Logical3OpInstruction(regb32, regb32, regb32, regb32, 0x80);
		// std::cout << test1->ToString() << std::endl;

	}
};

}
