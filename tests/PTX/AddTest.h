#pragma once

#include "Test.h"

#include "CUDA/Buffer.h"
#include "CUDA/Kernel.h"
#include "CUDA/KernelInvocation.h"
#include "CUDA/Module.h"

#include "PTX/Module.h"
#include "PTX/Type.h"
#include "PTX/Functions/Function.h"
#include "PTX/Functions/EntryFunction.h"
#include "PTX/Functions/DataFunction.h"
#include "PTX/Instructions/Arithmetic/AddInstruction.h"
#include "PTX/Instructions/Arithmetic/MultiplyWideInstruction.h"
#include "PTX/Instructions/ControlFlow/ReturnInstruction.h"
#include "PTX/Instructions/Data/ConvertToAddressInstruction.h"
#include "PTX/Instructions/Data/LoadInstruction.h"
#include "PTX/Instructions/Data/MoveInstruction.h"
#include "PTX/Instructions/Data/StoreInstruction.h"
#include "PTX/Operands/Adapters/PointerAdapter.h"
#include "PTX/Operands/Address/Address.h"
#include "PTX/Operands/Address/RegisterAddress.h"
#include "PTX/Operands/Address/MemoryAddress.h"
#include "PTX/Operands/Value.h"
#include "PTX/Operands/Variables/Variable.h"
#include "PTX/Operands/Variables/IndexedRegister.h"
#include "PTX/Operands/Variables/Register.h"
#include "PTX/Statements/Label.h"
#include "PTX/StateSpaces/AddressableSpace.h"
#include "PTX/StateSpaces/RegisterSpace.h"
#include "PTX/StateSpaces/PointerSpace.h"
#include "PTX/StateSpaces/SpecialRegisterSpace.h"
#include "PTX/StateSpaces/SpaceAdapter.h"

namespace Test
{

class AddTest : public Test
{
public:
	AddTest() {}
	
	void Execute()
	{
		PTX::Module module;

		module.SetVersion(6, 1);
		module.SetDeviceTarget("sm_61");
		module.SetAddressSize(PTX::Bits::Bits64);

		PTX::EntryFunction<PTX::Pointer64Space<PTX::Float64Type>> *function = new PTX::EntryFunction<PTX::Pointer64Space<PTX::Float64Type>>();
		function->SetName("AddTest");
		function->SetVisible(true);

		PTX::Pointer64Space<PTX::Float64Type> *parameterSpace = new PTX::Pointer64Space<PTX::Float64Type>("AddTest_0");
		function->SetParameters(parameterSpace);

		PTX::SpecialRegisterSpace<PTX::Vector4Type<PTX::UInt32Type>> *srtid = new PTX::SpecialRegisterSpace<PTX::Vector4Type<PTX::UInt32Type>>("%tid");
		PTX::RegisterSpace<PTX::UInt32Type> *r32 = new PTX::RegisterSpace<PTX::UInt32Type>("%r", 1);
		PTX::RegisterSpace<PTX::UInt64Type> *r64 = new PTX::RegisterSpace<PTX::UInt64Type>("%rd", 4);
		PTX::RegisterSpace<PTX::Float64Type> *f64 = new PTX::RegisterSpace<PTX::Float64Type>("%f", 2);

		PTX::ParameterVariable<PTX::Pointer64Type<PTX::Float64Type>> *parameter = parameterSpace->GetVariable("AddTest_0");

		PTX::Register<PTX::UInt32Type> *tidx = new PTX::IndexedRegister4<PTX::UInt32Type>(srtid->GetVariable("%tid"), PTX::VectorElement::X);
																		       
		PTX::Register<PTX::UInt32Type> *r0 = r32->GetVariable("%r", 0);

		PTX::Register<PTX::UInt64Type> *rd0 = r64->GetVariable("%rd", 0);
		PTX::Register<PTX::UInt64Type> *rd1 = r64->GetVariable("%rd", 1);
		PTX::Register<PTX::UInt64Type> *rd2 = r64->GetVariable("%rd", 2);
		PTX::Register<PTX::UInt64Type> *rd3 = r64->GetVariable("%rd", 3);

		PTX::Register<PTX::Float64Type> *f0 = f64->GetVariable("%f", 0);
		PTX::Register<PTX::Float64Type> *f1 = f64->GetVariable("%f", 1);

		PTX::Register<PTX::Pointer64Type<PTX::Float64Type>> *rd0_ptr = new PTX::Pointer64Adapter<PTX::Float64Type>(rd0);
		PTX::Register<PTX::Pointer64Type<PTX::Float64Type, PTX::AddressSpace::Global>> *rd1_ptr = new PTX::Pointer64Adapter<PTX::Float64Type, PTX::AddressSpace::Global>(rd1);

		PTX::Block *block = new PTX::Block();
		block->AddStatement(r32);
		block->AddStatement(r64); 
		block->AddStatement(f64); 

		block->AddStatement(new PTX::Load64Instruction<PTX::Pointer64Type<PTX::Float64Type>, PTX::AddressSpace::Param>(rd0_ptr, new PTX::MemoryAddress64<PTX::Pointer64Type<PTX::Float64Type>, PTX::AddressSpace::Param>(parameter)));
		block->AddStatement(new PTX::ConvertToAddress64Instruction<PTX::Float64Type, PTX::AddressSpace::Global>(rd1_ptr, rd0_ptr));
		block->AddStatement(new PTX::MoveInstruction<PTX::UInt32Type>(r0, tidx));
		block->AddStatement(new PTX::MultiplyWideInstruction<PTX::UInt64Type, PTX::UInt32Type>(rd2, r0, new PTX::UInt32Value(8)));
		block->AddStatement(new PTX::AddInstruction<PTX::UInt64Type>(rd3, rd1, rd2));

		block->AddStatement(new PTX::Load64Instruction<PTX::Float64Type, PTX::AddressSpace::Global>(f0, new PTX::RegisterAddress64<PTX::Float64Type, PTX::AddressSpace::Global>(rd1_ptr)));
		block->AddStatement(new PTX::AddInstruction<PTX::Float64Type>(f1, f0, new PTX::Float64Value(2)));

		PTX::Register<PTX::Pointer64Type<PTX::Float64Type, PTX::AddressSpace::Global>> *rd3_ptr = new PTX::Pointer64Adapter<PTX::Float64Type, PTX::AddressSpace::Global>(rd3);
		block->AddStatement(new PTX::Store64Instruction<PTX::Float64Type, PTX::AddressSpace::Global>(new PTX::RegisterAddress64<PTX::Float64Type, PTX::AddressSpace::Global>(rd3_ptr), f1));
		block->AddStatement(new PTX::ReturnInstruction());

		function->SetBody(block);

		module.AddFunction(function);
		std::string ptx = module.ToString();
		std::cout << ptx;

		CUDA::Module cModule(ptx);
		CUDA::Kernel kernel(function->GetName(), 1, cModule);

		size_t size = sizeof(double) * 1024;
		double *data = (double *)malloc(size);
		for (int i = 0; i < 1024; ++i)
		{
			data[i] = 1;
		}

		CUDA::Buffer buffer(data, size);
		buffer.AllocateOnGPU();
		buffer.TransferToGPU();

		CUDA::KernelInvocation invocation(kernel);
		invocation.SetBlockShape(1024, 1, 1);
		invocation.SetGridShape(1, 1, 1);
		invocation.SetParam(0, buffer);
		invocation.Launch();

		buffer.TransferToCPU();

		for (int i = 0; i < 1024; ++i)
		{
			if (data[i] == 3)
			{
				continue;
			}

			std::cerr << "[Error] Result incorrect at index " << i << " [" << data[i] << " != " << 3 << "]" << std::endl;
			return;
		}
		std::cerr << "[Info] Kernel execution successful" << std::endl;
	}
};

}
