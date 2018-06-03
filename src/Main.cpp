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
	PTX::Program *ptxProgram = codegen->GenerateProgram(program);
	std::cout << ptxProgram->ToString() << std::endl;
	std::exit(EXIT_SUCCESS);

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

	Test::ConditionalTest test;
	test.Execute();
}
