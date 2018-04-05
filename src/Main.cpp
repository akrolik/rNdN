#include <iostream>
#include <cstring>

#include "GPUUtil/CUDA.h"
#include "GPUUtil/CUDABuffer.h"
#include "GPUUtil/CUDAKernel.h"
#include "GPUUtil/CUDAKernelInvocation.h"
#include "GPUUtil/CUDAModule.h"

int main(int argc, char *argv[])
{
	if (sizeof(void *) == 4)
	{
		std::cerr << "[Error] 64-bit platform required" << std::endl;
		std::exit(EXIT_FAILURE);
	}

	CUDA c;
	c.Initialize();

	if (c.GetDeviceCount() == 0)
	{
		std::cerr << "[Error] No connected devices detected" << std::endl;
		std::exit(EXIT_FAILURE);
	}

	std::unique_ptr<CUDADevice>& device = c.GetDevice(0);
	device->SetActive();

	c.CreateContext(device);

	CUDAModule module;

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

	CUDAKernel kernel("_Z8myKernelPi", std::string(myPtx64), 1);
	module.AddKernel(kernel);

	size_t size = sizeof(int) * 1024;
        int *data = (int *)malloc(size);
	std::memset(data, 0, size);

	CUDABuffer buffer(data, size);
	buffer.AllocateOnGPU();
	buffer.TransferToGPU();

	CUDAKernelInvocation invocation(kernel);
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
