#pragma once

#include <cuda.h>
#include <cuda_runtime.h>
 
namespace CUDA {

class Buffer
{
public:
	Buffer(size_t size) : Buffer(nullptr, size) {}
	Buffer(void *buffer, size_t size) : m_CPUBuffer(buffer), m_size(size)
	{
		const auto multiple = 1024;
		m_paddedSize = (((m_size + multiple - 1) / multiple) * multiple);
	}

	void AllocateOnGPU();
	void Clear();
	void TransferToGPU();
	void TransferToCPU();

	void *GetCPUBuffer() const { return m_CPUBuffer; }
	CUdeviceptr& GetGPUBuffer() { return m_GPUBuffer; }

	void SetCPUBuffer(void *buffer) { m_CPUBuffer = buffer; }

	size_t GetSize() const { return m_size; }
	size_t GetPaddedSize() const { return m_paddedSize; }

private:
	void *m_CPUBuffer = nullptr;
	CUdeviceptr m_GPUBuffer;

	size_t m_size = 0;
	size_t m_paddedSize = 0;
};

}
