#include <iostream>
#include <cstring>
#include <chrono>

#include "Codegen/CodeGenerator.h"
#include "HorseIR/Tree/Program.h"
#include "PTX/Program.h"
#include "PTX/Type.h"

#include "CUDA/Buffer.h"
#include "CUDA/Kernel.h"
#include "CUDA/KernelInvocation.h"
#include "CUDA/libdevice.h"
#include "CUDA/Module.h"
#include "CUDA/Platform.h"
#include "CUDA/Utils.h"

#include "PTX/ArithmeticTest.h"
#include "PTX/ComparisonTest.h"
#include "PTX/ControlFlowTest.h"
#include "PTX/DataTest.h"
#include "PTX/LogicalTest.h"
#include "PTX/ShiftTest.h"
#include "PTX/SynchronizationTest.h"

// #include "PTX/AddTest.h"
// #include "PTX/BasicTest.h"
// #include "PTX/ConditionalTest.h"

int yyparse();

HorseIR::Program *program;

int main(int argc, char *argv[])
{
	// Disable cache for CUDA so compile times are accurate. In a production compiler
	// this would be turned off for efficiency

	setenv("CUDA_CACHE_DISABLE", "1", 1);

	if (sizeof(void *) == 4)
	{
		std::cerr << "[ERROR] 64-bit platform required" << std::endl;
		std::exit(EXIT_FAILURE);
	}

	// Initialize the CUDA platform and the device driver

	auto timeCUDA_start = std::chrono::steady_clock::now();

	CUDA::Platform p;
	p.Initialize();

	// Check to make sure there is at least one detected GPU

	if (p.GetDeviceCount() == 0)
	{
		std::cerr << "[ERROR] No connected devices detected" << std::endl;
		std::exit(EXIT_FAILURE);
	}

	// By default we use the first CUDA capable GPU for computations

	std::unique_ptr<CUDA::Device>& device = p.GetDevice(0);
	device->SetActive();

	// Complete the CUDA initialization by creating a CUDA context for the device

	p.CreateContext(device);

	auto timeCUDA_end = std::chrono::steady_clock::now();

	// Load the libdevice library from file, compile it to PTX, and generate a cubin
	// binary file. Doing so at runtime means we can support new versions of
	// libdevice and future compute versions
	//
	// The library will be linked later (if needed). We consider this cost a "setup"
	// cost that is not included in the compile time since the library must only be
	// compiled once

	auto timeLibrary_start = std::chrono::steady_clock::now();

	CUDA::ExternalModule libdevice = CUDA::libdevice::CreateModule(*device);

	auto timeLibrary_end = std::chrono::steady_clock::now();

	// Parse the input HorseIR program from stdin and generate an AST

	auto timeFrontend_start = std::chrono::steady_clock::now();

	yyparse();

	auto timeFrontend_end = std::chrono::steady_clock::now();

	std::cout << "[INFO] Input HorseIR program" << std::endl;
	std::cout << program->ToString() << std::endl;

	// Generate 64-bit PTX code from the input HorseIR for the current device

	auto timeCode_start = std::chrono::steady_clock::now();

	auto codegen = new Codegen::CodeGenerator<PTX::Bits::Bits64>(device->GetComputeCapability());
	PTX::Program *ptxProgram = codegen->Generate(program);

	auto timeCode_end = std::chrono::steady_clock::now();

	std::cout << "[INFO] Generated PTX program" << std::endl;
	for (const auto& module : ptxProgram->GetModules())
	{
		std::cout << module->ToString() << std::endl;
		// std::cout << module->ToJSON().dump(4) << std::endl;
	}
	std::cout << std::endl;

	// Generate the CUDA module for the program

	auto timeJIT_start = std::chrono::steady_clock::now();

	CUDA::Module cModule;
	for (const auto& module : ptxProgram->GetModules())
	{
		cModule.AddPTXModule(module->ToString());
	}
	cModule.AddLinkedModule(libdevice);
	cModule.Compile();

	auto timeJIT_end = std::chrono::steady_clock::now();

	size_t size = sizeof(long) * 100;
	long *dataA = (long *)malloc(size);
	// float *dataB = (float *)malloc(size);
	long *dataC = (long *)malloc(size);
	for (int i = 0; i < 100; ++i)
	{
		dataA[i] = -50 + i;//1 - float(i)/100;
		// dataB[i] = true;
		dataC[i] = 0;
	}

	// Fetch the handle to the entry function (always called main)

	auto timeExec_start = std::chrono::steady_clock::now();

	CUDA::Kernel kernel("main", 0, cModule);

	// Initialize buffers and kernel invocation

	CUDA::Buffer bufferA(dataA, size);
	// CUDA::Buffer bufferB(dataB, size);
	CUDA::Buffer bufferC(dataC, size);
	bufferA.AllocateOnGPU(); bufferA.TransferToGPU();
	// bufferB.AllocateOnGPU(); bufferB.TransferToGPU();
	bufferC.AllocateOnGPU();

	CUDA::KernelInvocation invocation(kernel);
	invocation.SetBlockShape(100, 1, 1);
	invocation.SetGridShape(1, 1, 1);
	invocation.SetParam(0, bufferA);
	// invocation.SetParam(1, bufferB);
	invocation.SetParam(1, bufferC);
	invocation.Launch();

	bufferC.TransferToCPU();

	auto timeExec_end = std::chrono::steady_clock::now();

	// Verify results

	for (int i = 0; i < 100; ++i)
	{
		// if (dataC[i] == 0.4794255386)
		// {
			// continue;
		// }

		std::cout << "[RESULT] signum(" << -50 + i << ") = " << dataC[i] << std::endl;
		// std::cout << "[RESULT] sin(" << 1 - float(i)/100 << ") = " << dataC[i] << std::endl;
		// std::cerr << "[ERROR] Result incorrect at index " << i << " [" << dataC[i] << " != " << 0.479426 << "]" << std::endl;
		// std::exit(EXIT_FAILURE);
	}

	auto cudaTime = std::chrono::duration_cast<std::chrono::microseconds>(timeCUDA_end - timeCUDA_start).count();
	auto libraryTime = std::chrono::duration_cast<std::chrono::microseconds>(timeLibrary_end - timeLibrary_start).count();
	auto frontendTime = std::chrono::duration_cast<std::chrono::microseconds>(timeFrontend_end - timeFrontend_start).count();
	auto codegenTime = std::chrono::duration_cast<std::chrono::microseconds>(timeCode_end - timeCode_start).count();
	auto jitTime = std::chrono::duration_cast<std::chrono::microseconds>(timeJIT_end - timeJIT_start).count();
	auto compileTime = std::chrono::duration_cast<std::chrono::microseconds>(timeJIT_end - timeFrontend_start).count();

	auto executionTime = std::chrono::duration_cast<std::chrono::microseconds>(timeExec_end - timeExec_start).count();

	std::cout << "[INFO] Kernel Execution Successful" << std::endl;
	std::cout << "[INFO] Pipeline Timings" << std::endl;
	std::cout << "         - CUDA Init: " << cudaTime << " mus" << std::endl;
	std::cout << "         - Libraries: " << libraryTime << " mus" << std::endl;
	std::cout << "         - Frontend: " << frontendTime << " mus" << std::endl;
	std::cout << "         - Codegen: " << codegenTime << " mus" << std::endl;
	std::cout << "         - PTX JIT: " << jitTime << " mus" << std::endl;
	std::cout << "[INFO] Total Compile Time: " << compileTime << " mus\n";
	std::cout << "[INFO] Execution Time: " << executionTime << " mus" << std::endl;
}
