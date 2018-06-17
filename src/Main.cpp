#include <iostream>
#include <cstring>
#include <chrono>

#include "Codegen/CodeGenerator.h"
#include "HorseIR/Tree/Program.h"
#include "PTX/Program.h"
#include "PTX/Type.h"

#include "CUDA/Platform.h"

#include "PTX/ArithmeticTest.h"
#include "PTX/ComparisonTest.h"
#include "PTX/ControlFlowTest.h"
#include "PTX/DataTest.h"
#include "PTX/LogicalTest.h"
#include "PTX/ShiftTest.h"

#include "PTX/AddTest.h"
#include "PTX/BasicTest.h"
#include "PTX/ConditionalTest.h"

int yyparse();

HorseIR::Program *program;

int main(int argc, char *argv[])
{
	setenv("CUDA_CACHE_DISABLE", "1", 1);

	auto cuda_begin = std::chrono::steady_clock::now();
	if (sizeof(void *) == 4)
	{
		std::cerr << "[ERROR] 64-bit platform required" << std::endl;
		std::exit(EXIT_FAILURE);
	}

	CUDA::Platform p;
	p.Initialize();

	if (p.GetDeviceCount() == 0)
	{
		std::cerr << "[ERROR] No connected devices detected" << std::endl;
		std::exit(EXIT_FAILURE);
	}

	std::unique_ptr<CUDA::Device>& device = p.GetDevice(0);
	device->SetActive();

	p.CreateContext(device);
	auto cuda_end = std::chrono::steady_clock::now();

	auto sp_begin = std::chrono::steady_clock::now();
	yyparse();
	auto sp_end = std::chrono::steady_clock::now();

	std::cout << program->ToString() << std::endl;

	auto code_begin = std::chrono::steady_clock::now();
	//TODO: sm_61 dynamic
	auto codegen = new CodeGenerator<PTX::Bits::Bits64>("sm_61");
	PTX::Program *ptxProgram = codegen->Generate(program);
	auto code_end = std::chrono::steady_clock::now();

	for (auto module : ptxProgram->GetModules())
	{
		std::cout << module->ToString() << std::endl;
	}

	auto jit_begin = std::chrono::steady_clock::now();
	CUDA::Module cModule(ptxProgram->GetModules()[0]->ToString());
	CUDA::Kernel kernel("main", 0, cModule);
	auto jit_end = std::chrono::steady_clock::now();

	size_t size = sizeof(int64_t) * 100;
	int64_t *data = (int64_t *)malloc(size);
	for (int i = 0; i < 100; ++i)
	{
		data[i] = 0;
	}

	auto exec_begin = std::chrono::steady_clock::now();
	CUDA::Buffer buffer(data, size);
	buffer.AllocateOnGPU();

	CUDA::KernelInvocation invocation(kernel);
	invocation.SetBlockShape(100, 1, 1);
	invocation.SetGridShape(1, 1, 1);
	invocation.SetParam(0, buffer);
	invocation.Launch();

	buffer.TransferToCPU();
	auto exec_end = std::chrono::steady_clock::now();

	for (int i = 0; i < 100; ++i)
	{
		if (data[i] == 3)
		{
			continue;
		}

		std::cerr << "[ERROR] Result incorrect at index " << i << " [" << data[i] << " != " << 3 << "]" << std::endl;
		std::exit(EXIT_FAILURE);
	}

	std::cout << "[INFO] Kernel execution successful" << std::endl;
	std::cout << "[Timings]" << std::endl;
	std::cout << "  CUDA Init: " << std::chrono::duration_cast<std::chrono::microseconds>(cuda_end - cuda_begin).count() << " mus\n";
	std::cout << "  Scan+Parse: " << std::chrono::duration_cast<std::chrono::microseconds>(sp_end - sp_begin).count() << " mus\n";
	std::cout << "  Codegen: " << std::chrono::duration_cast<std::chrono::microseconds>(code_end - code_begin).count() << " mus\n";
	std::cout << "  PTX JIT: " << std::chrono::duration_cast<std::chrono::microseconds>(jit_end - jit_begin).count() << " mus\n";
	std::cout << "    Total Compile Time: " << std::chrono::duration_cast<std::chrono::microseconds>(jit_end - sp_begin).count() << " mus\n";
	std::cout << "  Execution: " << std::chrono::duration_cast<std::chrono::microseconds>(exec_end - exec_begin).count() << " mus\n";
}
