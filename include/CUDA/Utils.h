#pragma once

#include <nvvm.h>
#include "nvPTXCompiler.h"

#include "Utils/Logger.h"

#define checkDriverResult(result) CUDA::_checkDriverResult(result, __FILE__, __LINE__)
#define checkRuntimeError(error) CUDA::_checkRuntimeError(error, __FILE__, __LINE__)
#define checkNVVMResult(result) CUDA::_checkNVVMResult(result, __FILE__, __LINE__)
#define checkNVPTXResult(result) CUDA::_checkNVPTXResult(result, __FILE__, __LINE__)

namespace CUDA {

static void _checkDriverResult(CUresult result, const char *file, int line)
{
	if (result == CUDA_SUCCESS)
	{
		return;
	}

	const char *name = nullptr;
	const char *string = nullptr;

	CUresult l_result = cuGetErrorName(result, &name);
	if (l_result == CUDA_ERROR_INVALID_VALUE || name == nullptr)
	{
		Utils::Logger::LogError("Unknown CUDA error <" + std::string(file) + ":" + std::to_string(line) + ">", "DRIVER ERROR");
	}

	cuGetErrorString(result, &string);

	Utils::Logger::LogErrorPart(std::string(name) + " <" + std::string(file) + ":" + std::to_string(line) + ">", "DRIVER ERROR");
	Utils::Logger::LogError(string, "DRIVER ERROR");
}

static void _checkRuntimeError(cudaError_t error, const char *file, int line)
{
	if (error == cudaSuccess)
	{
		return;
	}

	Utils::Logger::LogError(std::string(cudaGetErrorString(error)) + " <" + std::string(file) + ":" + std::to_string(line) + ">", "RUNTIME ERROR");
}

static void _checkNVVMResult(nvvmResult result, const char *file, int line)
{
	if (result == NVVM_SUCCESS)
	{
		return;
	}

	Utils::Logger::LogError(std::string(nvvmGetErrorString(result)) + " <" + std::string(file) + ":" + std::to_string(line) + ">", "NVVM ERROR");
}

static void _checkNVPTXResult(nvPTXCompileResult result, const char *file, int line)
{
	if (result == NVPTXCOMPILE_SUCCESS)
	{
		return;
	}


	std::string error;
	switch (result)
	{
#define NVPTXCASE(c) case c: error = #c; break;

		NVPTXCASE(NVPTXCOMPILE_SUCCESS)
		NVPTXCASE(NVPTXCOMPILE_ERROR_INVALID_COMPILER_HANDLE)
		NVPTXCASE(NVPTXCOMPILE_ERROR_INVALID_INPUT)
		NVPTXCASE(NVPTXCOMPILE_ERROR_COMPILATION_FAILURE)
		NVPTXCASE(NVPTXCOMPILE_ERROR_INTERNAL)
		NVPTXCASE(NVPTXCOMPILE_ERROR_OUT_OF_MEMORY)
		NVPTXCASE(NVPTXCOMPILE_ERROR_COMPILER_INVOCATION_INCOMPLETE)
		NVPTXCASE(NVPTXCOMPILE_ERROR_UNSUPPORTED_PTX_VERSION)
	}
	Utils::Logger::LogError(error + " <" + std::string(file) + ":" + std::to_string(line) + ">", "NVPTX ERROR");
}

static void Synchronize()
{
	checkRuntimeError(cudaDeviceSynchronize());
}

}
