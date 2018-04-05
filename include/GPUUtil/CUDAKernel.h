#ifndef R3D3_GPUUTIL_CUDAKERNEL
#define R3D3_GPUUTIL_CUDAKERNEL

#include <cuda.h>
#include <cuda_runtime.h>

#include <string>

class CUDAKernel
{
public:
	CUDAKernel(std::string name, std::string ptx, unsigned int paramCount);

	std::string GetName() { return m_name; }
	unsigned int GetParametersCount() { return m_parametersCount; }
	void *GetBinary() { return m_binary; }

	CUfunction& GetKernel() { return m_kernel; }

private:
	std::string m_name;
	std::string m_ptx;
	unsigned int m_parametersCount;

	void *m_binary = nullptr;
	size_t m_binarySize = 0;

	CUfunction m_kernel;

	void Compile();
};

#endif
