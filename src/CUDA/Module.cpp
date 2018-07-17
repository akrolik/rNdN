#include "CUDA/Module.h"

#include "CUDA/Utils.h"

#include <chrono>
#include <cstring>
#include <functional>
#include <iostream>

namespace CUDA {

void Module::AddLinkedModule(const ExternalModule& module)
{
	m_linkedModules.push_back(std::cref(module));
}

void Module::AddPTXModule(const std::string& code)
{
	m_code.push_back(code);
}

void Module::Compile()
{
	// Initialize the JIT options for the compile

	CUjit_option optionKeys[6];
	void *optionVals[6];
	unsigned int optionCount = 6;

	float l_wallTime;
	char l_infoLog[8192];
	char l_errorLog[8192];
	unsigned int l_logSize = 8192;

	optionKeys[0] = CU_JIT_WALL_TIME;
	optionVals[0] = (void *)&l_wallTime;

	optionKeys[1] = CU_JIT_INFO_LOG_BUFFER;
	optionVals[1] = (void *)l_infoLog;

	optionKeys[2] = CU_JIT_INFO_LOG_BUFFER_SIZE_BYTES;
	optionVals[2] = (void *)(long)l_logSize;

	optionKeys[3] = CU_JIT_ERROR_LOG_BUFFER;
	optionVals[3] = (void *)l_errorLog;

	optionKeys[4] = CU_JIT_ERROR_LOG_BUFFER_SIZE_BYTES;
	optionVals[4] = (void *)(long)l_logSize;

	optionKeys[5] = CU_JIT_LOG_VERBOSE;
	optionVals[5] = (void *)true;

	// Create the linker with the options above

	CUlinkState linkerState;
	checkDriverResult(cuLinkCreate(optionCount, optionKeys, optionVals, &linkerState));

	// Add the PTX source code to the linker for compilation. Note that this will invoke the compile
	// despite the name only pertaining to the linker

	auto timeCompile_start = std::chrono::steady_clock::now();

	for (const auto& code : m_code)
	{
		CUresult result = cuLinkAddData(linkerState, CU_JIT_INPUT_PTX, (void *)code.c_str(), code.length() + 1, "PTX Module", 0, nullptr, nullptr);
		if (result != CUDA_SUCCESS)
		{
			std::cerr << "[ERROR] PTX failed to compile" << std::endl << l_errorLog << std::endl;
			checkDriverResult(result);
		}
	}

	auto timeCompile_end = std::chrono::steady_clock::now();

	// Add the external libraries to the linker (if any)

	auto timeLibraries_start = std::chrono::steady_clock::now();

	for (const auto& module : m_linkedModules)
	{
		CUresult result = cuLinkAddData(linkerState, CU_JIT_INPUT_CUBIN, module.get().GetBinary(), module.get().GetBinarySize(), "Library Module", 0, nullptr, nullptr);
		if (result != CUDA_SUCCESS)
		{
			std::cerr << "[ERROR] Library cubin image failed to load" << std::endl << l_errorLog << std::endl;
			checkDriverResult(result);
		}
	}

	auto timeLibraries_end = std::chrono::steady_clock::now();

	// Create the binary for the module, containing all code from the kernels as well as the
	// exrenral libraries

	auto timeLink_start = std::chrono::steady_clock::now();

	void *binary = nullptr;
	size_t binarySize = 0;
	checkDriverResult(cuLinkComplete(linkerState, &binary, &binarySize));

	m_binary = malloc(binarySize);
	m_binarySize = binarySize;
	std::memcpy(m_binary, binary, binarySize);

	auto timeLink_end = std::chrono::steady_clock::now();

	auto compileTime = std::chrono::duration_cast<std::chrono::microseconds>(timeCompile_end - timeCompile_start).count();
	auto librariesTime = std::chrono::duration_cast<std::chrono::microseconds>(timeLibraries_end - timeLibraries_start).count();
	auto linkTime = std::chrono::duration_cast<std::chrono::microseconds>(timeLink_end - timeLink_start).count();

	std::cout << "[INFO] PTX compiled in " << compileTime + librariesTime + linkTime << " mus" << std::endl;
	std::cout << "         - Compilation: " << compileTime << " mus" << std::endl;
	std::cout << "         - Libraries: " << librariesTime << " mus" << std::endl;
	std::cout << "         - Link: " << linkTime << " mus" << std::endl;
	std::cout << l_infoLog << std::endl;

	// Load the binary into the module and cleanup the linker session

	checkDriverResult(cuModuleLoadData(&m_module, m_binary));
	checkDriverResult(cuLinkDestroy(linkerState));
}

}
