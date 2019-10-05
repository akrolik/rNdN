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

	void *GetCPUBuffer() const { return m_CPUBuffer; }
	CUdeviceptr& GetGPUBuffer() { return m_GPUBuffer; }

	void SetCPUBuffer(void *buffer) { m_CPUBuffer = buffer; }

	size_t GetSize() const { return m_size; }
	size_t GetPaddedSize() const
	{
		const auto multiple = 1024;
		return (((m_size + multiple - 1) / multiple) * multiple);
	}

private:
	void *m_CPUBuffer = nullptr;
	CUdeviceptr m_GPUBuffer;

	size_t m_size = 0;
};

}
