#pragma once

#include <cuda.h>
#include <cuda_runtime.h>
 
namespace CUDA {

class Buffer
{
public:
	Buffer(void *buffer, size_t size) : m_CPUBuffer(buffer), m_size(size) {}

	void AllocateOnGPU();
	void TransferToGPU();
	void TransferToCPU();

	void *GetCPUBuffer() { return m_CPUBuffer; }
	CUdeviceptr& GetGPUBuffer() { return m_GPUBuffer; }
	size_t GetSize() { return m_size; }

private:
	void *m_CPUBuffer = nullptr;
	CUdeviceptr m_GPUBuffer;
	size_t m_size = 0;
};

}
