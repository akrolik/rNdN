#include <iostream>
#include <cstring>

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
	yyparse();
	std::cout << program->ToString() << std::endl;

	CodeGenerator *codegen = new CodeGenerator("sm_61", PTX::Bits::Bits64);
	PTX::Program *ptxProgram = codegen->Generate(program);
	for (auto module : ptxProgram->GetModules())
	{
		std::cout << module->ToString() << std::endl;
	}

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

	CUDA::Module cModule(ptxProgram->GetModules()[0]->ToString());
	CUDA::Kernel kernel("main", 0, cModule);

	size_t size = sizeof(int64_t) * 100;
	int64_t *data = (int64_t *)malloc(size);
	for (int i = 0; i < 100; ++i)
	{
		data[i] = 0;
	}

	CUDA::Buffer buffer(data, size);
	buffer.AllocateOnGPU();

	CUDA::KernelInvocation invocation(kernel);
	invocation.SetBlockShape(100, 1, 1);
	invocation.SetGridShape(1, 1, 1);
	invocation.SetParam(0, buffer);
	invocation.Launch();

	buffer.TransferToCPU();
	for (int i = 0; i < 100; ++i)
	{
		if (data[i] == 3)
		{
			continue;
		}

		std::cerr << "[Error] Result incorrect at index " << i << " [" << data[i] << " != " << 3 << "]" << std::endl;
		std::exit(EXIT_FAILURE);
	}

	std::cerr << "[Info] Kernel execution successful" << std::endl;
	std::exit(EXIT_SUCCESS);
}
