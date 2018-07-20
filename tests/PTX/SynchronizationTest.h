#pragma once

#include "Test.h"

#include "PTX/Type.h"
#include "PTX/StateSpace.h"
#include "PTX/Declarations/VariableDeclaration.h"

#include "PTX/Instructions/Synchronization/BarrierInstruction.h"
#include "PTX/Instructions/Synchronization/BarrierReductionInstruction.h"

namespace Test
{

class SynchronizationTest : public Test
{
public:
	void Execute()
	{
		PTX::RegisterDeclaration<PTX::UInt32Type> *u32 = new PTX::RegisterDeclaration<PTX::UInt32Type>("%u", 4);
		PTX::RegisterDeclaration<PTX::PredicateType> *p = new PTX::RegisterDeclaration<PTX::PredicateType>("%p", 4);

		const PTX::Register<PTX::UInt32Type> *reg_u32 = u32->GetVariable("%u", 0);
		const PTX::Register<PTX::PredicateType> *reg_p = p->GetVariable("%p", 0);

		PTX::BarrierInstruction *test1 = new PTX::BarrierInstruction(reg_u32, reg_u32, false);
		test1->SetAligned(true);
		std::cout << test1->ToString(0) << std::endl;

		PTX::BarrierReductionInstruction<PTX::PredicateType> *test2 = new PTX::BarrierReductionInstruction<PTX::PredicateType>(reg_p, reg_u32, reg_u32, PTX::BarrierReductionInstruction<PTX::PredicateType>::Operation::And, reg_p);
		std::cout << test2->ToString(0) << std::endl;

		PTX::BarrierReductionInstruction<PTX::UInt32Type> *test3 = new PTX::BarrierReductionInstruction<PTX::UInt32Type>(reg_u32, reg_u32, reg_u32, reg_p);
		std::cout << test3->ToString(0) << std::endl;
	}
};

}
