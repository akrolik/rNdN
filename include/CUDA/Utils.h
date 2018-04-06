#pragma once

#include <iostream>

#define checkDriverResult(result) CUDA::_checkDriverResult(result, __FILE__, __LINE__)
#define checkRuntimeError(error) CUDA::_checkRuntimeError(error, __FILE__, __LINE__)

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
	if (l_result == CUDA_ERROR_INVALID_VALUE)
	{
		std::cerr << "[Driver Error] Unknown CUDA error <" << file << ":" << line << ">" << std::endl;
		std::exit(EXIT_FAILURE);
	}

	cuGetErrorString(result, &string);

	std::cerr << "[Driver Error] " << name << " <" << file << ":" << line << ">" << std::endl << string << std::endl;
	std::exit(EXIT_FAILURE);
}

static void _checkRuntimeError(cudaError_t error, const char *file, int line)
{
	if (error == cudaSuccess)
	{
		return;
	}

	std::cerr << "[Runtime Error] " << cudaGetErrorString(error) << " <" << file << ":" << line << ">" << std::endl;
	std::exit(EXIT_FAILURE);

}
}
