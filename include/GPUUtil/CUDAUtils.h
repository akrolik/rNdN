#ifndef R3D3_GPUUTIL_CUDAUTILS
#define R3D3_GPUUTIL_CUDAUTILS

#include "GPUUtil/CUDA.h"

#define checkDriverResult(result) CUDA::_checkDriverResult(result, __FILE__, __LINE__)
#define checkRuntimeError(error) CUDA::_checkRuntimeError(error, __FILE__, __LINE__)

#endif
