#pragma once

#include "Test.h"

#include "PTX/Instructions/AddInstruction.h"
#include "PTX/Instructions/SubtractInstruction.h"
#include "PTX/Instructions/MultiplyInstruction.h"
#include "PTX/Instructions/MultiplyWideInstruction.h"
#include "PTX/Instructions/MADInstruction.h"
#include "PTX/Instructions/MADWideInstruction.h"
#include "PTX/Instructions/SADInstruction.h"
#include "PTX/Instructions/DivideInstruction.h"
#include "PTX/Instructions/RemainderInstruction.h"
#include "PTX/Instructions/AbsoluteInstruction.h"
#include "PTX/Instructions/NegateInstruction.h"
#include "PTX/Instructions/MinimumInstruction.h"
#include "PTX/Instructions/MaximumInstruction.h"
#include "PTX/Instructions/FMAInstruction.h"
#include "PTX/Instructions/ReciprocalInstruction.h"
#include "PTX/Instructions/RootInstruction.h"
#include "PTX/Instructions/ReciprocalRootInstruction.h"
#include "PTX/Instructions/SineInstruction.h"
#include "PTX/Instructions/CosineInstruction.h"
#include "PTX/Instructions/Log2Instruction.h"
#include "PTX/Instructions/Exp2Instruction.h"

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
