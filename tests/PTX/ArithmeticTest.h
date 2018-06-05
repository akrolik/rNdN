#pragma once

#include "Test.h"

#include "PTX/Type.h"
#include "PTX/StateSpace.h"
#include "PTX/Declarations/VariableDeclaration.h"

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
		PTX::RegisterDeclaration<PTX::Int32Type> *s32 = new PTX::RegisterDeclaration<PTX::Int32Type>("%r", 4);
		PTX::RegisterDeclaration<PTX::Float32Type> *f32 = new PTX::RegisterDeclaration<PTX::Float32Type>("%f", 4);
		PTX::RegisterDeclaration<PTX::Float64Type> *f64 = new PTX::RegisterDeclaration<PTX::Float64Type>("%fd", 4);

		PTX::Register<PTX::Int32Type> *reg_s32 = s32->GetVariable("%r", 0);
		PTX::Register<PTX::Float32Type> *reg_f32 = f32->GetVariable("%f", 0);
		PTX::Register<PTX::Float64Type> *reg_f64 = f64->GetVariable("%fd", 0);

		PTX::AddInstruction<PTX::Int32Type> *test1 = new PTX::AddInstruction<PTX::Int32Type>(reg_s32, reg_s32, reg_s32);
		std::cout << test1->ToString() << std::endl;

		PTX::AbsoluteInstruction<PTX::Float32Type> *test2 = new PTX::AbsoluteInstruction<PTX::Float32Type>(reg_f32, reg_f32);
		test2->SetFlushSubnormal(true);
		std::cout << test2->ToString() << std::endl;

		PTX::AbsoluteInstruction<PTX::Float64Type> *test3 = new PTX::AbsoluteInstruction<PTX::Float64Type>(reg_f64, reg_f64);
		std::cout << test3->ToString() << std::endl;

		PTX::AddInstruction<PTX::Float32Type> *test4 = new PTX::AddInstruction<PTX::Float32Type>(reg_f32, reg_f32, reg_f32);
		test4->SetRoundingMode(PTX::Float32Type::RoundingMode::Nearest);
		test4->SetFlushSubnormal(true);
		test4->SetSaturate(true);
		std::cout << test4->ToString() << std::endl;

		PTX::ReciprocalRootInstruction<PTX::Float64Type> *test5 = new PTX::ReciprocalRootInstruction<PTX::Float64Type>(reg_f64, reg_f64);
		test5->SetFlushSubnormal(true);
		std::cout << test5->ToString() << std::endl;
	}
};

}
