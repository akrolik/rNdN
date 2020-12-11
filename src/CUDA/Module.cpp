#include "CUDA/Module.h"

#include "CUDA/Utils.h"

#include "Utils/Chrono.h"
#include "Utils/Logger.h"
#include "Utils/Options.h"

#include <cstring>
#include <functional>

namespace CUDA {

void Module::AddExternalModule(const ExternalModule& module)
{
	m_externalModules.push_back(std::cref(module));
}

void Module::AddELFModule(const Assembler::ELFBinary& module)
{
	m_elfModules.push_back(std::cref(module));
}

void Module::AddPTXModule(const std::string& code)
{
	m_ptxModules.push_back(code);
}

void Module::Compile()
{
	// Initialize the JIT options for the compile

	auto timeCreate_start = Utils::Chrono::Start("Create");

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

	Utils::Chrono::End(timeCreate_start);

	// Add the PTX source code to the linker for compilation. Note that this will invoke the compiler
	// despite the name only pertaining to the linker

	auto timeAssemble_start = Utils::Chrono::Start("Assemble");

	for (const auto& code : m_ptxModules)
	{
		auto result = cuLinkAddData(linkerState, CU_JIT_INPUT_PTX, (void *)code.c_str(), code.length() + 1, "PTX Module", 0, nullptr, nullptr);
		if (result != CUDA_SUCCESS)
		{
			Utils::Logger::LogErrorPart("PTX failed to compile", Utils::Logger::ErrorPrefix);
			Utils::Logger::LogErrorPart(l_errorLog, Utils::Logger::NoPrefix);
			checkDriverResult(result);
		}
	}

	// Add relocatable ELF files to linker

	for (const auto& module : m_elfModules)
	{
		auto result = cuLinkAddData(linkerState, CU_JIT_INPUT_CUBIN, module.get().GetData(), module.get().GetSize(), "ELF Module", 0, nullptr, nullptr);
		if (result != CUDA_SUCCESS)
		{
			Utils::Logger::LogErrorPart("ELF failed to load", Utils::Logger::ErrorPrefix);
			Utils::Logger::LogErrorPart(l_errorLog, Utils::Logger::NoPrefix);
			checkDriverResult(result);
		}

	}

	Utils::Chrono::End(timeAssemble_start);

	// Add the external libraries to the linker (if any)

	auto timeLibraries_start = Utils::Chrono::Start("Libraries");

	for (const auto& module : m_externalModules)
	{
		CUresult result = cuLinkAddData(linkerState, CU_JIT_INPUT_CUBIN, module.get().GetBinary(), module.get().GetBinarySize(), "Library Module", 0, nullptr, nullptr);
		if (result != CUDA_SUCCESS)
		{
			Utils::Logger::LogErrorPart("Library cubin image failed to load", Utils::Logger::ErrorPrefix);
			Utils::Logger::LogErrorPart(l_errorLog, Utils::Logger::NoPrefix);
			checkDriverResult(result);
		}
	}

	Utils::Chrono::End(timeLibraries_start);

	// Create the binary for the module, containing all code from the kernels as well as the external libraries

	auto timeLink_start = Utils::Chrono::Start("Link");

	void *binary = nullptr;
	size_t binarySize = 0;
	checkDriverResult(cuLinkComplete(linkerState, &binary, &binarySize));

	m_binary = malloc(binarySize);
	m_binarySize = binarySize;
	std::memcpy(m_binary, binary, binarySize);

	Utils::Chrono::End(timeLink_start);

	// Load the binary into the module and cleanup the linker session

	auto timeLoad_start = Utils::Chrono::Start("Load");

	checkDriverResult(cuModuleLoadData(&m_module, m_binary));
	checkDriverResult(cuLinkDestroy(linkerState));

	Utils::Chrono::End(timeLoad_start);

	// Log compilation info to stdout

	if (Utils::Options::IsDebug_Print())
	{
		Utils::Logger::LogDebug(l_infoLog);
	}
}

}
