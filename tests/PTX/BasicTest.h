#pragma once

#include "Test.h"

#include "CUDA/Buffer.h"
#include "CUDA/Kernel.h"
#include "CUDA/KernelInvocation.h"
#include "CUDA/Module.h"

#include "PTX/Module.h"
#include "PTX/Type.h"
#include "PTX/StateSpace.h"
#include "PTX/Functions/Function.h"
#include "PTX/Functions/DataFunction.h"
#include "PTX/Instructions/Arithmetic/AddInstruction.h"
#include "PTX/Instructions/Arithmetic/MADInstruction.h"
#include "PTX/Instructions/Arithmetic/MultiplyWideInstruction.h"
#include "PTX/Instructions/ControlFlow/ReturnInstruction.h"
#include "PTX/Instructions/Data/ConvertToAddressInstruction.h"
#include "PTX/Instructions/Data/LoadInstruction.h"
#include "PTX/Instructions/Data/MoveInstruction.h"
#include "PTX/Instructions/Data/StoreInstruction.h"
#include "PTX/Operands/Adapters/PointerAdapter.h"
#include "PTX/Operands/Adapters/SignedAdapter.h"
#include "PTX/Operands/Adapters/UnsignedAdapter.h"
#include "PTX/Operands/Address/Address.h"
#include "PTX/Operands/Address/RegisterAddress.h"
#include "PTX/Operands/Address/MemoryAddress.h"
#include "PTX/Operands/Value.h"
#include "PTX/Operands/Variables/Variable.h"
#include "PTX/Operands/Variables/IndexedRegister.h"
#include "PTX/Operands/Variables/Register.h"

namespace Test
{

class BasicTest : public Test
{
public:
	BasicTest() {}
	
	void Execute()
	{
		PTX::Module module;

		module.SetVersion(6, 1);
		module.SetDeviceTarget("sm_61");
		module.SetAddressSize(PTX::Bits::Bits64);

		PTX::DataFunction<PTX::VoidType(PTX::ParameterVariable<PTX::Pointer64Type<PTX::UInt64Type>>)> *function = new PTX::DataFunction<PTX::VoidType(PTX::ParameterVariable<PTX::Pointer64Type<PTX::UInt64Type>>)>();
		function->SetName("BasicTest");
		function->SetEntry(true);
		function->SetLinkDirective(PTX::Declaration::LinkDirective::Visible);

		PTX::Pointer64Declaration<PTX::UInt64Type> *parameterDeclaration = new PTX::Pointer64Declaration<PTX::UInt64Type>("BasicTest_0");
		function->SetParameters(parameterDeclaration);

		PTX::SpecialRegisterDeclaration<PTX::Vector4Type<PTX::UInt32Type>> *srtid = new PTX::SpecialRegisterDeclaration<PTX::Vector4Type<PTX::UInt32Type>>("%tid");
		PTX::SpecialRegisterDeclaration<PTX::Vector4Type<PTX::UInt32Type>> *srntid = new PTX::SpecialRegisterDeclaration<PTX::Vector4Type<PTX::UInt32Type>>("%ntid");
		PTX::SpecialRegisterDeclaration<PTX::Vector4Type<PTX::UInt32Type>> *srctaid = new PTX::SpecialRegisterDeclaration<PTX::Vector4Type<PTX::UInt32Type>>("%ctaid");
		PTX::RegisterDeclaration<PTX::Int32Type> *r32 = new PTX::RegisterDeclaration<PTX::Int32Type>("%r", 5);
		PTX::RegisterDeclaration<PTX::Int64Type> *r64 = new PTX::RegisterDeclaration<PTX::Int64Type>("%rd", 4);

		PTX::ParameterVariable<PTX::Pointer64Type<PTX::UInt64Type>> *parameter = parameterDeclaration->GetVariable("BasicTest_0");

		PTX::Register<PTX::UInt32Type> *tidx = new PTX::IndexedRegister4<PTX::UInt32Type>(srtid->GetVariable("%tid"), PTX::VectorElement::X);
		PTX::Register<PTX::UInt32Type> *ntidx = new PTX::IndexedRegister4<PTX::UInt32Type>(srntid->GetVariable("%ntid"), PTX::VectorElement::X);
		PTX::Register<PTX::UInt32Type> *ctaidx = new PTX::IndexedRegister4<PTX::UInt32Type>(srctaid->GetVariable("%ctaid"), PTX::VectorElement::X);
																		       
		PTX::Register<PTX::Int32Type> *r0 = r32->GetVariable("%r", 0);
		PTX::Register<PTX::Int32Type> *r1 = r32->GetVariable("%r", 1);
		PTX::Register<PTX::Int32Type> *r2 = r32->GetVariable("%r", 2);
		PTX::Register<PTX::Int32Type> *r3 = r32->GetVariable("%r", 3);
		PTX::Register<PTX::Int32Type> *r4 = r32->GetVariable("%r", 4);

		PTX::Register<PTX::Int64Type> *rd0 = r64->GetVariable("%rd", 0);
		PTX::Register<PTX::Int64Type> *rd1 = r64->GetVariable("%rd", 1);
		PTX::Register<PTX::Int64Type> *rd2 = r64->GetVariable("%rd", 2);
		PTX::Register<PTX::Int64Type> *rd3 = r64->GetVariable("%rd", 3);

		PTX::Register<PTX::Pointer64Type<PTX::UInt64Type>> *rd0_ptr = new PTX::Pointer64Adapter<PTX::UInt64Type>(new PTX::Unsigned64Adapter(rd0));
		PTX::Register<PTX::Pointer64Type<PTX::UInt64Type, PTX::GlobalSpace>> *rd1_ptr = new PTX::Pointer64Adapter<PTX::UInt64Type, PTX::GlobalSpace>(new PTX::Unsigned64Adapter(rd1));

		function->AddStatement(r32);
		function->AddStatement(r64); 

		function->AddStatement(new PTX::Load64Instruction<PTX::Pointer64Type<PTX::UInt64Type>, PTX::ParameterSpace>(rd0_ptr, new PTX::MemoryAddress64<PTX::Pointer64Type<PTX::UInt64Type>, PTX::ParameterSpace>(parameter)));
		function->AddStatement(new PTX::ConvertToAddress64Instruction<PTX::UInt64Type, PTX::GlobalSpace>(rd1_ptr, rd0_ptr));
		function->AddStatement(new PTX::MoveInstruction<PTX::UInt32Type>(new PTX::Unsigned32Adapter(r0), ntidx));
		function->AddStatement(new PTX::MoveInstruction<PTX::UInt32Type>(new PTX::Unsigned32Adapter(r1), ctaidx));
		function->AddStatement(new PTX::MoveInstruction<PTX::UInt32Type>(new PTX::Unsigned32Adapter(r2), tidx));

		PTX::MADInstruction<PTX::Int32Type> *madInstruction = new PTX::MADInstruction<PTX::Int32Type>(r3, r0, r1, r2);
		madInstruction->SetLower(true);
		function->AddStatement(madInstruction);

		function->AddStatement(new PTX::MultiplyWideInstruction<PTX::Int64Type, PTX::Int32Type>(rd2, r3, new PTX::Int32Value(4)));
		function->AddStatement(new PTX::AddInstruction<PTX::Int64Type>(rd3, rd1, rd2));
		function->AddStatement(new PTX::AddInstruction<PTX::UInt32Type>(new PTX::Unsigned32Adapter(r4), new PTX::Unsigned32Adapter(r3), new PTX::UInt32Value(1)));

		PTX::Register<PTX::Pointer64Type<PTX::UInt32Type, PTX::GlobalSpace>> *rd3_ptr = new PTX::Pointer64Adapter<PTX::UInt32Type, PTX::GlobalSpace>(new PTX::Unsigned64Adapter(rd3));
		function->AddStatement(new PTX::Store64Instruction<PTX::UInt32Type, PTX::GlobalSpace>(new PTX::RegisterAddress64<PTX::UInt32Type, PTX::GlobalSpace>(rd3_ptr), new PTX::Unsigned32Adapter(r4)));
		function->AddStatement(new PTX::ReturnInstruction());

		module.AddDeclaration(function);
		std::string ptx = module.ToString();
		std::cout << ptx;

		CUDA::Module cModule(ptx);
		CUDA::Kernel kernel(function->GetName(), 1, cModule);

		size_t size = sizeof(int) * 1024;
		int *data = (int *)malloc(size);
		std::memset(data, 0, size);

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
			if (data[i] == i + 1)
			{
				continue;
			}
			std::cerr << "[Error] Result incorrect at index " << i << " [" << data[i] << " != " << i + 1 << "]" << std::endl;
			return;
		}
		std::cout << "[Info] Kernel execution successful" << std::endl;
	}
};

}
