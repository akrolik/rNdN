#include <iostream>
#include <cstring>

#include "CUDA/Platform.h"

#include "PTX/ArithmeticTest.h"
#include "PTX/AddTest.h"
#include "PTX/BasicTest.h"
#include "PTX/ConditionalTest.h"

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

	Test::AddTest test;
	// Test::BasicTest test;
	// Test::ConditionalTest test;
	test.Execute();
}
