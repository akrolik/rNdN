#pragma once

#include "Test.h"

#include "PTX/Instructions/AddInstruction.h"
#include "PTX/Instructions/SubtractInstruction.h"
#include "PTX/Instructions/MultiplyInstruction.h"
#include "PTX/Instructions/MultiplyWideInstruction.h"
#include "PTX/Instructions/MADInstruction.h"
#include "PTX/Instructions/MADWideInstruction.h"
#include "PTX/Instructions/SADInstruction.h"
// #include "PTX/Instructions/DivInstruction.h"
// #include "PTX/Instructions/RemInstruction.h"
// #include "PTX/Instructions/AbsInstruction.h"
// #include "PTX/Instructions/NegInstruction.h"
// #include "PTX/Instructions/MinInstruction.h"
// #include "PTX/Instructions/MaxInstruction.h"
// #include "PTX/Instructions/FmaInstruction.h"
// #include "PTX/Instructions/RcpInstruction.h"
// #include "PTX/Instructions/SqrtInstruction.h"
// #include "PTX/Instructions/RSqrtInstruction.h"
// #include "PTX/Instructions/SinInstruction.h"
// #include "PTX/Instructions/CosInstruction.h"
// #include "PTX/Instructions/Log2Instruction.h"
// #include "PTX/Instructions/Exp2Instruction.h"

namespace Test
{

class ArithmeticTest : public Test
{
public:
	void Execute()
	{
		// PTX::MadInstruction<PTX::Int16Type> *mad1 = new PTX::MadInstruction<PTX::Int16Type>(nullptr, nullptr, nullptr, nullptr);
		// PTX::MadInstruction<PTX::Int8Type> *mad2 = new PTX::MadInstruction<PTX::Int8Type>(nullptr, nullptr, nullptr, nullptr);
		// PTX::MadInstruction<PTX::IntType<PTX::Bits::Bits8>> *mad3 = new PTX::MadInstruction<PTX::IntType<PTX::Bits::Bits8>>(nullptr, nullptr, nullptr, nullptr);

		// PTX::SADInstruction<PTX::Int16Type> *sad1 = new PTX::SADInstruction<PTX::Int16Type>(nullptr, nullptr, nullptr);
		// PTX::SADInstruction<PTX::Float32Type> *sad2 = new PTX::SADInstruction<PTX::Float32Type>(nullptr, nullptr, nullptr);
	}
};

}
