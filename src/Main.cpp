#include <iostream>
#include <cstring>

#include "CUDA/Buffer.h"
#include "CUDA/Kernel.h"
#include "CUDA/KernelInvocation.h"
#include "CUDA/Module.h"
#include "CUDA/Platform.h"

#include "PTX/Module.h"
#include "PTX/Type.h"
#include "PTX/Function.h"
#include "PTX/Functions/EntryFunction.h"
#include "PTX/Functions/DataFunction.h"
#include "PTX/Operands/Register.h"
#include "PTX/Operands/IndexedRegister.h"
#include "PTX/Statements/LoadInstruction.h"
#include "PTX/Statements/MoveInstruction.h"
#include "PTX/StateSpaces/ParameterSpace.h"
#include "PTX/StateSpaces/RegisterSpace.h"

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
	module.SetAddressSize(PTX::Module::AddressSize64);

	PTX::EntryFunction<PTX::ParameterSpace<PTX::Type::UInt64>> *function = new PTX::EntryFunction<PTX::ParameterSpace<PTX::Type::UInt64>>();
	function->SetName("_Z8myKernelPi");
	function->SetVisible(true);

	PTX::ParameterSpace<PTX::Type::UInt64> *parameter = new PTX::ParameterSpace<PTX::Type::UInt64>(PTX::ParameterSpace<PTX::Type::UInt64>::GenericSpace, "_Z8myKernelPi_param_0");
	function->SetParameters(parameter);

	PTX::Block *block = new PTX::Block();

	PTX::RegisterSpace<PTX::Type::UInt32> *r32 = new PTX::RegisterSpace<PTX::Type::UInt32>("r", 5);
	PTX::RegisterSpace<PTX::Type::UInt64> *r64 = new PTX::RegisterSpace<PTX::Type::UInt64>("rd", 4);
	block->AddStatement(r32);
	block->AddStatement(r64);

	PTX::Register<PTX::Type::UInt32> *r1 = r32->GetRegister(0);
	PTX::Register<PTX::Type::UInt32> *r2 = r32->GetRegister(1);
	PTX::Register<PTX::Type::UInt32> *r3 = r32->GetRegister(2);
	PTX::Register<PTX::Type::UInt32> *r4 = r32->GetRegister(3);
	PTX::Register<PTX::Type::UInt32> *r5 = r32->GetRegister(4);

	PTX::Register<PTX::Type::UInt64> *rd1 = r64->GetRegister(0);
	PTX::Register<PTX::Type::UInt64> *rd2 = r64->GetRegister(1);
	PTX::Register<PTX::Type::UInt64> *rd3 = r64->GetRegister(2);
	PTX::Register<PTX::Type::UInt64> *rd4 = r64->GetRegister(3);

	PTX::Address<PTX::Type::UInt64> *address = new PTX::Address<PTX::Type::UInt64>(parameter);
	PTX::LoadInstruction<PTX::Type::UInt64> *ld = new PTX::LoadInstruction<PTX::Type::UInt64>(rd1, address);
	block->AddStatement(ld);

	PTX::RegisterSpace<PTX::Type::UInt32, PTX::VectorSize::Vector4> *sregntid = new PTX::RegisterSpace<PTX::Type::UInt32, PTX::VectorSize::Vector4>({"ntid"});
	PTX::IndexedRegister<PTX::Type::UInt32, PTX::VectorSize::Vector4> *ntidx = sregntid->GetRegister(0, PTX::VectorElement::X);
	// PTX::IndexedRegister<PTX::Type::UInt32, PTX::VectorSize::Vector4> *ntidx = new PTX::IndexedRegister<PTX::Type::UInt32, PTX::VectorSize::Vector4>(sregntid, 0, PTX::VectorElement::X);
	PTX::MoveInstruction<PTX::Type::UInt32> *mv = new PTX::MoveInstruction<PTX::Type::UInt32>(r1, ntidx);
	block->AddStatement(mv);

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
