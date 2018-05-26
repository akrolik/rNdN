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
		PTX::RegisterSpace<PTX::Int32Type> *s32 = new PTX::RegisterSpace<PTX::Int32Type>("%r", 4);
		PTX::Register<PTX::Int32Type> *regs32 = s32->GetVariable("%r", 0);

		PTX::AddInstruction<PTX::Int32Type> *test0 = new PTX::AddInstruction<PTX::Int32Type>(regs32, regs32, regs32);
		std::cout << test0->ToString() << std::endl;

		PTX::RegisterSpace<PTX::Float32Type> *f32 = new PTX::RegisterSpace<PTX::Float32Type>("%f", 4);
		PTX::Register<PTX::Float32Type> *reg32 = f32->GetVariable("%f", 0);

		PTX::AbsoluteInstruction<PTX::Float32Type> *test1 = new PTX::AbsoluteInstruction<PTX::Float32Type>(reg32, reg32);
		test1->SetFlushSubnormal(true);
		std::cout << test1->ToString() << std::endl;

		PTX::RegisterSpace<PTX::Float64Type> *f64 = new PTX::RegisterSpace<PTX::Float64Type>("%fd", 4);
		PTX::Register<PTX::Float64Type> *reg64 = f64->GetVariable("%fd", 0);

		PTX::AbsoluteInstruction<PTX::Float64Type> *test2 = new PTX::AbsoluteInstruction<PTX::Float64Type>(reg64, reg64);
		// test2->SetFlushSubnormal(true);
		std::cout << test2->ToString() << std::endl;

		PTX::AddInstruction<PTX::Float32Type> *test3 = new PTX::AddInstruction<PTX::Float32Type>(reg32, reg32, reg32);
		test3->SetRoundingMode(PTX::Float32Type::RoundingMode::Nearest);
		test3->SetFlushSubnormal(true);
		test3->SetSaturate(true);
		std::cout << test3->ToString() << std::endl;

		PTX::ReciprocalRootInstruction<PTX::Float64Type> *test4 = new PTX::ReciprocalRootInstruction<PTX::Float64Type>(reg64, reg64);
		test4->SetFlushSubnormal(true);
		std::cout << test4->ToString() << std::endl;

		// PTX::SADInstruction<PTX::Float16x2Type> *test2 = new PTX::SADInstruction<PTX::Float16x2Type>(nullptr, nullptr, nullptr);
		// PTX::MadInstruction<PTX::Int16Type> *mad1 = new PTX::MadInstruction<PTX::Int16Type>(nullptr, nullptr, nullptr, nullptr);
		// PTX::MadInstruction<PTX::Int8Type> *mad2 = new PTX::MadInstruction<PTX::Int8Type>(nullptr, nullptr, nullptr, nullptr);
		// PTX::MadInstruction<PTX::IntType<PTX::Bits::Bits8>> *mad3 = new PTX::MadInstruction<PTX::IntType<PTX::Bits::Bits8>>(nullptr, nullptr, nullptr, nullptr);

		// PTX::SADInstruction<PTX::Int16Type> *sad1 = new PTX::SADInstruction<PTX::Int16Type>(nullptr, nullptr, nullptr);
		// PTX::SADInstruction<PTX::Float32Type> *sad2 = new PTX::SADInstruction<PTX::Float32Type>(nullptr, nullptr, nullptr);
	}
};

}
