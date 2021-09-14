#include "CUDA/Compiler.h"

#include "CUDA/Utils.h"

#include "Utils/Chrono.h"
#include "Utils/Logger.h"
#include "Utils/Options.h"

namespace CUDA {

void Compiler::AddExternalModule(const ExternalModule& module)
{
	m_externalModules.push_back(std::cref(module));
}

void Compiler::AddELFModule(const Assembler::ELFBinary& module)
{
	m_elfModules.push_back(std::cref(module));
}

void Compiler::AddPTXModule(const std::string& code)
{
	m_ptxModules.push_back(code);
}

void Compiler::AddFileModule(const std::string& file)
{
	m_fileModules.push_back(file);
}

void Compiler::Compile(const std::unique_ptr<Device>& device)
{
	auto timeAssembler_start = Utils::Chrono::Start("CUDA assembler");

	// Initialize the JIT options for the compile

	auto timeCreate_start = Utils::Chrono::Start("Create");

	constexpr unsigned int optionCount = 8;
	CUjit_option optionKeys[optionCount];
	void *optionVals[optionCount];

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

	optionKeys[6] = CU_JIT_FAST_COMPILE;
	optionVals[6] = (void *)true;

	optionKeys[7] = CU_JIT_OPTIMIZATION_LEVEL;
	optionVals[7] = (void *)(long)Utils::Options::GetOptimize_PtxasLevel();

	// Create the linker with the options above

	CUlinkState linkerState;
	checkDriverResult(cuLinkCreate(optionCount, optionKeys, optionVals, &linkerState));

	Utils::Chrono::End(timeCreate_start);

	auto timeAssemble_start = Utils::Chrono::Start("Assemble");

	if (m_ptxModules.size() > 0)
	{
		if (Utils::Options::IsDebug_Print())
		{
			unsigned int minorVer, majorVer;
			checkNVPTXResult(nvPTXCompilerGetVersion(&majorVer, &minorVer));

			Utils::Logger::LogDebug("nvPTX compiler version: " + std::to_string(majorVer) + "." + std::to_string(minorVer));
		}
	}

	// Add the PTX source code to the linker for compilation. Note that this will invoke the compiler
	// despite the name only pertaining to the linker

	for (const auto& code : m_ptxModules)
	{
		auto sm = "--gpu-name=" + device->GetComputeCapability();
		const char* compileOptions[] = { sm.c_str(), "--compile-only" };

		nvPTXCompilerHandle compiler = NULL;
		checkNVPTXResult(nvPTXCompilerCreate(&compiler, code.length(), code.c_str()));

		auto status = nvPTXCompilerCompile(compiler, 2, compileOptions);
		if (status != NVPTXCOMPILE_SUCCESS)
		{
			size_t errorSize;
			checkNVPTXResult(nvPTXCompilerGetErrorLogSize(compiler, &errorSize));

			if (errorSize != 0)
			{
				auto errorLog = new char[errorSize + 1];
				checkNVPTXResult(nvPTXCompilerGetErrorLog(compiler, errorLog));

				Utils::Logger::LogErrorPart("PTX failed to compile", Utils::Logger::ErrorPrefix);
				Utils::Logger::LogErrorPart(errorLog, Utils::Logger::NoPrefix);

				delete[] errorLog;
			}
			checkNVPTXResult(status);
		}

		size_t elfSize;
		checkNVPTXResult(nvPTXCompilerGetCompiledProgramSize(compiler, &elfSize));

		auto elf = malloc(elfSize);
		checkNVPTXResult(nvPTXCompilerGetCompiledProgram(compiler, (void*)elf));

		if (Utils::Options::IsDebug_Print())
		{
			size_t infoSize;
			checkNVPTXResult(nvPTXCompilerGetInfoLogSize(compiler, &infoSize));

			if (infoSize != 0)
			{
				auto infoLog = new char[infoSize + 1];
				checkNVPTXResult(nvPTXCompilerGetInfoLog(compiler, infoLog));

				Utils::Logger::LogDebug(infoLog);

				delete[] infoLog;
			}
		}

		checkNVPTXResult(nvPTXCompilerDestroy(&compiler));

		auto result = cuLinkAddData(linkerState, CU_JIT_INPUT_CUBIN, elf, elfSize, "PTX Module", 0, nullptr, nullptr);
		if (result != CUDA_SUCCESS)
		{
			Utils::Logger::LogErrorPart("PTX failed to compile", Utils::Logger::ErrorPrefix);
			Utils::Logger::LogErrorPart(l_errorLog, Utils::Logger::NoPrefix);
			checkDriverResult(result);
		}
	}

	// Add files to the linker

	for (const auto& file : m_fileModules)
	{
		auto result = cuLinkAddFile(linkerState, CU_JIT_INPUT_CUBIN, file.c_str(), 0, nullptr, nullptr);
		if (result != CUDA_SUCCESS)
		{
			Utils::Logger::LogErrorPart("ELF file failed to load", Utils::Logger::ErrorPrefix);
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

	checkDriverResult(cuLinkDestroy(linkerState));

	// Log compilation info to stdout

	if (Utils::Options::IsDebug_Print())
	{
		Utils::Logger::LogDebug(l_infoLog);
	}

	Utils::Chrono::End(timeAssembler_start);
}

}
