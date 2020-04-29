#pragma once

#include <cuda.h>
#include <cuda_runtime.h>
#include <string>

#include "CUDA/Data.h"
 
namespace CUDA {

class Buffer : public Data
{
public:
	static void Copy(Buffer *destination, Buffer *source, size_t size, size_t destinationOffset = 0, size_t sourceOffset = 0);

	Buffer(size_t size) : Buffer(nullptr, size) {}
	Buffer(void *buffer, size_t size);

	~Buffer();

	// Transfers

	void AllocateOnGPU();
	void Clear();
	void TransferToGPU();
	void TransferToCPU();

	// Buffers

	void *GetCPUBuffer() const { return m_CPUBuffer; }
	void SetCPUBuffer(void *buffer) { m_CPUBuffer = buffer; }
	bool HasCPUBuffer () const { return (m_CPUBuffer != nullptr); }

	CUdeviceptr& GetGPUBuffer() { return m_GPUBuffer; }
	void *GetAddress() override { return &m_GPUBuffer; }

	// Sizes

	size_t GetSize() const { return m_size; }
	void SetSize(size_t size) { m_size = size; }

	size_t GetAllocatedSize() const { return m_allocatedSize; }

	// Tag

	const std::string& GetTag() const { return m_tag; }
	void SetTag(const std::string &tag) { m_tag = tag; }

private:
	std::string ChronoDescription(const std::string& operation, size_t size);

	void *m_CPUBuffer = nullptr;
	CUdeviceptr m_GPUBuffer;

	size_t m_size = 0;
	size_t m_allocatedSize = 0;

	std::string m_tag;
};

}
