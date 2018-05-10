#include <iostream>
#include <cstring>

#include "CUDA/Buffer.h"
#include "CUDA/Kernel.h"
#include "CUDA/KernelInvocation.h"
#include "CUDA/Module.h"
#include "CUDA/Platform.h"

#include "PTX/Module.h"
#include "PTX/Type.h"
#include "PTX/Functions/Function.h"
#include "PTX/Functions/EntryFunction.h"
#include "PTX/Functions/DataFunction.h"
#include "PTX/Operands/Adapters/SignedAdapter.h"
#include "PTX/Operands/Adapters/UnsignedAdapter.h"
#include "PTX/Operands/Address/Address.h"
#include "PTX/Operands/Address/RegisterAddress.h"
#include "PTX/Operands/Address/VariableAddress.h"
#include "PTX/Operands/Register/Register.h"
#include "PTX/Operands/Register/IndexedRegister.h"
#include "PTX/Operands/Value/Int32Value.h"
#include "PTX/Operands/Value/UInt32Value.h"
#include "PTX/Operands/ZeroExtendAddress.h"
#include "PTX/Operands/ZeroExtendRegister.h"
#include "PTX/Instructions/AddInstruction.h"
#include "PTX/Instructions/ConvertAddressInstruction.h"
#include "PTX/Instructions/LoadInstruction.h"
#include "PTX/Instructions/MadInstruction.h"
#include "PTX/Instructions/MoveInstruction.h"
#include "PTX/Instructions/MulWideInstruction.h"
#include "PTX/Instructions/ReturnInstruction.h"
#include "PTX/Instructions/StoreInstruction.h"
#include "PTX/StateSpaces/ParameterSpace.h"
#include "PTX/StateSpaces/RegisterSpace.h"
#include "PTX/StateSpaces/SpecialRegisterSpace.h"

int yyparse();

int main(int argc, char *argv[])
{
	// yyparse();

	if (sizeof(void *) == 4)
	{
		std::cerr << "[Error] 64-bit platform required" << std::endl;
		std::exit(EXIT_FAILURE);
	}

	CUDA::Platform p;
	p.Initialize();

	if (p.GetDeviceCount() == 0)
	{
		std::cerr << "[Error] No connected devices detected" << std::endl;
		std::exit(EXIT_FAILURE);
	}

	std::unique_ptr<CUDA::Device>& device = p.GetDevice(0);
	device->SetActive();

	p.CreateContext(device);

	PTX::Module module;

	module.SetVersion(3, 2);
	module.SetDeviceTarget("sm_20");
	module.SetAddressSize(PTX::Bits::Bits64);

	PTX::EntryFunction<PTX::ParameterSpace<PTX::UInt64Type>> *function = new PTX::EntryFunction<PTX::ParameterSpace<PTX::UInt64Type>>();
	function->SetName("_Z8myKernelPi");
	function->SetVisible(true);

	PTX::ParameterSpace<PTX::UInt64Type> *parameterSpace = new PTX::ParameterSpace<PTX::UInt64Type>("_Z8myKernelPi_param_0");
	function->SetParameters(parameterSpace);

	PTX::Variable<PTX::UInt64Type> *parameter = parameterSpace->GetVariable("_Z8myKernelPi_param_0");

	PTX::Block *block = new PTX::Block();

	PTX::SpecialRegisterSpace<PTX::UInt32Type, PTX::VectorSize::Vector4> *srtid = new PTX::SpecialRegisterSpace<PTX::UInt32Type, PTX::VectorSize::Vector4>("%tid");
	PTX::SpecialRegisterSpace<PTX::UInt32Type, PTX::VectorSize::Vector4> *srntid = new PTX::SpecialRegisterSpace<PTX::UInt32Type, PTX::VectorSize::Vector4>("%ntid");
	PTX::SpecialRegisterSpace<PTX::UInt32Type, PTX::VectorSize::Vector4> *srctaid = new PTX::SpecialRegisterSpace<PTX::UInt32Type, PTX::VectorSize::Vector4>("%ctaid");
	PTX::RegisterSpace<PTX::Int32Type> *r32 = new PTX::RegisterSpace<PTX::Int32Type>("%r", 5);
	PTX::RegisterSpace<PTX::Int64Type> *r64 = new PTX::RegisterSpace<PTX::Int64Type>("%rd", 4);
	block->AddStatement(r32);
	block->AddStatement(r64);

	PTX::IndexedRegister<PTX::UInt32Type, PTX::VectorSize::Vector4> *tidx = srtid->GetRegister("%tid", 0, PTX::VectorElement::X);
	PTX::IndexedRegister<PTX::UInt32Type, PTX::VectorSize::Vector4> *ntidx = srntid->GetRegister("%ntid", 0, PTX::VectorElement::X);
	PTX::IndexedRegister<PTX::UInt32Type, PTX::VectorSize::Vector4> *ctaidx = srctaid->GetRegister("%ctaid", 0, PTX::VectorElement::X);

	PTX::Register<PTX::Int32Type> *r1 = r32->GetRegister("%r", 0);
	PTX::Register<PTX::Int32Type> *r2 = r32->GetRegister("%r", 1);
	PTX::Register<PTX::Int32Type> *r3 = r32->GetRegister("%r", 2);
	PTX::Register<PTX::Int32Type> *r4 = r32->GetRegister("%r", 3);
	PTX::Register<PTX::Int32Type> *r5 = r32->GetRegister("%r", 4);

	PTX::Register<PTX::Int64Type> *rd1 = r64->GetRegister("%rd", 0);
	PTX::Register<PTX::Int64Type> *rd2 = r64->GetRegister("%rd", 1);
	PTX::Register<PTX::Int64Type> *rd3 = r64->GetRegister("%rd", 2);
	PTX::Register<PTX::Int64Type> *rd4 = r64->GetRegister("%rd", 3);

	PTX::VariableAddress<PTX::Bits::Bits64, PTX::UInt64Type> *address1 = new PTX::VariableAddress<PTX::Bits::Bits64, PTX::UInt64Type>(parameter);
	block->AddStatement(new PTX::LoadInstruction<PTX::Bits::Bits64, PTX::UInt64Type>(new PTX::UnsignedAdapter<PTX::Bits::Bits64>(rd1), address1));
	block->AddStatement(new PTX::ConvertAddressInstruction<PTX::Bits::Bits64>(new PTX::UnsignedAdapter<PTX::Bits::Bits64>(rd2), new PTX::UnsignedAdapter<PTX::Bits::Bits64>(rd1)));
	block->AddStatement(new PTX::MoveInstruction<PTX::UInt32Type>(new PTX::UnsignedAdapter<PTX::Bits::Bits32>(r1), ntidx));
	block->AddStatement(new PTX::MoveInstruction<PTX::UInt32Type>(new PTX::UnsignedAdapter<PTX::Bits::Bits32>(r2), ctaidx));
	block->AddStatement(new PTX::MoveInstruction<PTX::UInt32Type>(new PTX::UnsignedAdapter<PTX::Bits::Bits32>(r3), tidx));

	PTX::MadInstruction<PTX::Int32Type> *madInstruction = new PTX::MadInstruction<PTX::Int32Type>(r4, r1, r2, r3);
	madInstruction->SetLower(true);
	block->AddStatement(madInstruction);

	block->AddStatement(new PTX::MulWideInstruction<PTX::Int64Type, PTX::Int32Type>(rd3, r4, new PTX::Int32Value(4)));
	block->AddStatement(new PTX::AddInstruction<PTX::Int64Type>(rd4, rd2, rd3));
	block->AddStatement(new PTX::AddInstruction<PTX::UInt32Type>(new PTX::UnsignedAdapter<PTX::Bits::Bits32>(r5), new PTX::UnsignedAdapter<PTX::Bits::Bits32>(r4), new PTX::UInt32Value(1)));

	PTX::AddressRegister<PTX::Bits::Bits64> *addressReg = new PTX::AddressRegister<PTX::Bits::Bits64>(new PTX::UnsignedAdapter<PTX::Bits::Bits64>(rd4), PTX::AddressSpace::Global);
	PTX::RegisterAddress<PTX::Bits::Bits64, PTX::UInt64Type> *address2 = new PTX::RegisterAddress<PTX::Bits::Bits64, PTX::UInt64Type>(addressReg);

	PTX::Address<PTX::Bits::Bits64, PTX::UInt32Type> *address3 = new PTX::ZeroExtendAddress<PTX::Bits::Bits64, PTX::UInt32Type, PTX::UInt64Type>(address2);
	block->AddStatement(new PTX::StoreInstruction<PTX::Bits::Bits64, PTX::UInt32Type>(address3, new PTX::UnsignedAdapter<PTX::Bits::Bits32>(r5)));
	block->AddStatement(new PTX::ReturnInstruction());

        /*
	char myPtx64[] = "\n\
	.version 3.2\n\
	.target sm_20\n\
	.address_size 64\n\
	.visible .entry _Z8myKernelPi(\n\
		.param .u64 _Z8myKernelPi_param_0\n\
	)\n\
	{\n\
		.reg .s32 	%r<6>;\n\
		.reg .s64 	%rd<5>;\n\
		ld.param.u64 	%rd1, [_Z8myKernelPi_param_0];\n\
		cvta.to.global.u64 	%rd2, %rd1;\n\
		.loc 1 3 1\n\
		mov.u32 	%r1, %ntid.x;\n\
		mov.u32 	%r2, %ctaid.x;\n\
		mov.u32 	%r3, %tid.x;\n\
		mad.lo.s32 	%r4, %r1, %r2, %r3;\n\
		mul.wide.s32 	%rd3, %r4, 4;\n\
		add.s64 	%rd4, %rd2, %rd3;\n\
		.loc 1 4 1\n\
		add.u32         %r5, %r4, 1;\n\
		st.global.u32 	[%rd4], %r5;\n\
		.loc 1 5 2\n\
		ret;\n\
	}\n\
	";
	*/

	char bodyText[] = "\
		.reg .s32 	%r<6>;\n\
		.reg .s64 	%rd<5>;\n\
		ld.param.u64 	%rd1, [_Z8myKernelPi_param_0];\n\
		cvta.to.global.u64 	%rd2, %rd1;\n\
		mov.u32 	%r1, %ntid.x;\n\
		mov.u32 	%r2, %ctaid.x;\n\
		mov.u32 	%r3, %tid.x;\n\
		mad.lo.s32 	%r4, %r1, %r2, %r3;\n\
		mul.wide.s32 	%rd3, %r4, 4;\n\
		add.s64 	%rd4, %rd2, %rd3;\n\
		add.u32         %r5, %r4, 1;\n\
	st.global.u32 	[%rd4], %r5;\n\
		ret;\
	";

	function->SetBody(block);

	module.AddFunction(function);

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

	bool success = true;
	for (int i = 0; i < 1024; ++i)
	{
		if (data[i] == i + 1)
		{
			continue;
		}

		success = false;
		std::cerr << "[Error] Result incorrect at index " << i << " [" << data[i] << " != " << i << "]" << std::endl;
		break;
	}
	if (success)
	{
		std::cerr << "[Info] Kernel execution successful" << std::endl;
	}
}
