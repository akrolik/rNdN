#pragma once

#include "Test.h"

#include "CUDA/Module.h"

#include "PTX/Module.h"
#include "PTX/Block.h"
#include "PTX/Functions/EntryFunction.h"

#include "PTX/StateSpaces/AddressableSpace.h"
#include "PTX/StateSpaces/RegisterSpace.h"
#include "PTX/Operands/Variables/AddressableVariable.h"

#include "PTX/Instructions/ControlFlow/BranchInstruction.h"
#include "PTX/Instructions/ControlFlow/BranchIndexInstruction.h"
#include "PTX/Instructions/ControlFlow/CallInstruction.h"
#include "PTX/Instructions/ControlFlow/ExitInstruction.h"
#include "PTX/Instructions/ControlFlow/ReturnInstruction.h"

namespace Test {

class ControlFlowTest : public Test
{
public:
	void Execute()
	{
		PTX::Module module;
		module.SetVersion(6, 1);
		module.SetDeviceTarget("sm_61");
		module.SetAddressSize(PTX::Bits::Bits64);

		PTX::RegisterSpace<PTX::Bit32Type> *param_b32 = new PTX::RegisterSpace<PTX::Bit32Type>("%b");
		PTX::RegisterSpace<PTX::Int32Type> *param_s32 = new PTX::RegisterSpace<PTX::Int32Type>("%s");
		PTX::ParameterSpace<PTX::UInt64Type> *param_u64 = new PTX::ParameterSpace<PTX::UInt64Type>("%u");

		auto deviceFunction = new PTX::DataFunction<PTX::RegisterSpace<PTX::Bit32Type>, PTX::RegisterSpace<PTX::Int32Type>, PTX::ParameterSpace<PTX::UInt64Type>>();
		deviceFunction->SetName("device_function");
		deviceFunction->SetReturnSpace(param_b32);
		deviceFunction->SetParameters(param_s32, param_u64);

		module.AddFunction(deviceFunction);

		PTX::EntryFunction<> *entryFunction = new PTX::EntryFunction<>();
		entryFunction->SetName("CallTest");
		entryFunction->SetVisible(true);

		PTX::Block *block = new PTX::Block();

		PTX::RegisterSpace<PTX::Bit32Type> *b32 = new PTX::RegisterSpace<PTX::Bit32Type>("%b");
		PTX::RegisterSpace<PTX::Int32Type> *s32 = new PTX::RegisterSpace<PTX::Int32Type>("%s");
		PTX::ParameterSpace<PTX::UInt64Type> *u64 = new PTX::ParameterSpace<PTX::UInt64Type>("%u");

		block->AddStatement(b32);
		block->AddStatement(s32);
		block->AddStatement(u64);

		PTX::Register<PTX::Bit32Type> *regb32 = b32->GetVariable("%b");
		PTX::Register<PTX::Int32Type> *regs32 = s32->GetVariable("%s");
		PTX::ParameterVariable<PTX::UInt64Type> *varu64 = u64->GetVariable("%u");

		block->AddStatement(new PTX::CallInstruction<PTX::RegisterSpace<PTX::Bit32Type>, PTX::RegisterSpace<PTX::Int32Type>, PTX::ParameterSpace<PTX::UInt64Type>>(deviceFunction, regb32, regs32, varu64));

		entryFunction->SetBody(block);
		
		module.AddFunction(entryFunction);

		std::string ptx = module.ToString();
		std::cout << ptx;
		CUDA::Module cModule(ptx);
	}
};

}
