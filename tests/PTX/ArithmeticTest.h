#pragma once

#include "Test.h"

#include "PTX/Instructions/Arithmetic/AbsoluteInstruction.h"
#include "PTX/Instructions/Arithmetic/AddInstruction.h"
#include "PTX/Instructions/Arithmetic/BitFieldExtractInstruction.h"
#include "PTX/Instructions/Arithmetic/BitFieldInsertInstruction.h"
#include "PTX/Instructions/Arithmetic/BitFindInstruction.h"
#include "PTX/Instructions/Arithmetic/BitReverseInstruction.h"
#include "PTX/Instructions/Arithmetic/CopySignInstruction.h"
#include "PTX/Instructions/Arithmetic/CosineInstruction.h"
#include "PTX/Instructions/Arithmetic/CountLeadingZerosInstruction.h"
#include "PTX/Instructions/Arithmetic/DivideInstruction.h"
#include "PTX/Instructions/Arithmetic/Exp2Instruction.h"
#include "PTX/Instructions/Arithmetic/FindNthBitInstruction.h"
#include "PTX/Instructions/Arithmetic/FMAInstruction.h"
#include "PTX/Instructions/Arithmetic/Log2Instruction.h"
#include "PTX/Instructions/Arithmetic/MADInstruction.h"
#include "PTX/Instructions/Arithmetic/MADWideInstruction.h"
#include "PTX/Instructions/Arithmetic/MaximumInstruction.h"
#include "PTX/Instructions/Arithmetic/MinimumInstruction.h"
#include "PTX/Instructions/Arithmetic/MultiplyInstruction.h"
#include "PTX/Instructions/Arithmetic/MultiplyWideInstruction.h"
#include "PTX/Instructions/Arithmetic/NegateInstruction.h"
#include "PTX/Instructions/Arithmetic/PopulationCountInstruction.h"
#include "PTX/Instructions/Arithmetic/ReciprocalInstruction.h"
#include "PTX/Instructions/Arithmetic/ReciprocalRootInstruction.h"
#include "PTX/Instructions/Arithmetic/RemainderInstruction.h"
#include "PTX/Instructions/Arithmetic/RootInstruction.h"
#include "PTX/Instructions/Arithmetic/SADInstruction.h"
#include "PTX/Instructions/Arithmetic/SineInstruction.h"
#include "PTX/Instructions/Arithmetic/SubtractInstruction.h"
#include "PTX/Instructions/Arithmetic/TestPropertyInstruction.h"

namespace Test
{

class ArithmeticTest : public Test
{
public:
	void Execute()
	{
		PTX::AddInstruction<PTX::Float16x2Type> *test = new PTX::AddInstruction<PTX::Float16x2Type>(nullptr, nullptr, nullptr);
		test->SetSaturate(true);

		// PTX::SADInstruction<PTX::Float16x2Type> *test2 = new PTX::SADInstruction<PTX::Float16x2Type>(nullptr, nullptr, nullptr);
		// PTX::MadInstruction<PTX::Int16Type> *mad1 = new PTX::MadInstruction<PTX::Int16Type>(nullptr, nullptr, nullptr, nullptr);
		// PTX::MadInstruction<PTX::Int8Type> *mad2 = new PTX::MadInstruction<PTX::Int8Type>(nullptr, nullptr, nullptr, nullptr);
		// PTX::MadInstruction<PTX::IntType<PTX::Bits::Bits8>> *mad3 = new PTX::MadInstruction<PTX::IntType<PTX::Bits::Bits8>>(nullptr, nullptr, nullptr, nullptr);

		// PTX::SADInstruction<PTX::Int16Type> *sad1 = new PTX::SADInstruction<PTX::Int16Type>(nullptr, nullptr, nullptr);
		// PTX::SADInstruction<PTX::Float32Type> *sad2 = new PTX::SADInstruction<PTX::Float32Type>(nullptr, nullptr, nullptr);
	}
};

}
