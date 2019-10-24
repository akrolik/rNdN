#pragma once

#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>

#include "CUDA/Buffer.h"
#include "CUDA/Constant.h"
#include "CUDA/Kernel.h"

namespace CUDA {

class KernelInvocation
{
public:
	KernelInvocation(Kernel& kernel);

	void SetBlockShape(unsigned int x, unsigned int y, unsigned int z)
	{
		m_blockX = x;
		m_blockY = y;
		m_blockZ = z;
	}

	void SetGridShape(unsigned int x, unsigned int y, unsigned int z)
	{
		m_gridX = x;
		m_gridY = y;
		m_gridZ = z;
	}

	void AddParameter(Constant &value) { SetParameter(m_paramIndex++, value); }
	void AddParameter(Buffer &buffer) { SetParameter(m_paramIndex++, buffer); }

	void SetParameter(unsigned int index, Constant &value);
	void SetParameter(unsigned int index, Buffer &buffer);
	void SetSharedMemorySize(unsigned int bytes) { m_sharedMemorySize = bytes; }

	void Launch();

private:
	Kernel& m_kernel;

	unsigned int m_blockX = 0;
	unsigned int m_blockY = 0;
	unsigned int m_blockZ = 0;

	unsigned int m_gridX = 0;
	unsigned int m_gridY = 0;
	unsigned int m_gridZ = 0;

	unsigned int m_sharedMemorySize = 0;
	unsigned int m_paramIndex = 0;

	std::vector<void *> m_parameters;
};

}
