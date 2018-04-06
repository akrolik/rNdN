#pragma once

#include <cuda.h>
#include <cuda_runtime.h>

#include "CUDA/Buffer.h"
#include "CUDA/Kernel.h"

namespace CUDA {

class KernelInvocation
{
public:
	KernelInvocation(Kernel& kernel);
	~KernelInvocation();

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

	void SetParam(unsigned int index, Buffer &buffer);
	void Launch();

private:
	Kernel& m_kernel;

	unsigned int m_shapeX = 0;
	unsigned int m_shapeY = 0;
	unsigned int m_shapeZ = 0;

	unsigned int m_blocksX = 0;
	unsigned int m_blocksY = 0;
	unsigned int m_blocksZ = 0;

	void *m_parameters = nullptr;
};

}
