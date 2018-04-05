#ifndef R3D3_GPUUTIL_CUDAKERNELINVOCATION
#define R3D3_GPUUTIL_CUDAKERNELINVOCATION

#include <cuda.h>
#include <cuda_runtime.h>

#include "GPUUtil/CUDABuffer.h"
#include "GPUUtil/CUDAKernel.h"

class CUDAKernelInvocation
{
public:
	CUDAKernelInvocation(CUDAKernel& kernel);
	~CUDAKernelInvocation();

	void SetBlockShape(unsigned int x, unsigned int y, unsigned int z)
	{
		m_shapeX = x;
		m_shapeY = y;
		m_shapeZ = z;
	}

	void SetGridShape(unsigned int x, unsigned int y, unsigned int z)
	{
		m_blocksX = x;
		m_blocksY = y;
		m_blocksZ = z;
	}

	void SetParam(unsigned int index, CUDABuffer &buffer);
	void Launch();

private:
	CUDAKernel& m_kernel;

	unsigned int m_shapeX = 0;
	unsigned int m_shapeY = 0;
	unsigned int m_shapeZ = 0;

	unsigned int m_blocksX = 0;
	unsigned int m_blocksY = 0;
	unsigned int m_blocksZ = 0;

	void *m_parameters = nullptr;
};

#endif
