#pragma once

#include <cuda.h>
#include <cuda_runtime.h>
#include <string>

#include "CUDA/Data.h"
#include "CUDA/ConstantBuffer.h"

namespace CUDA {

class Buffer : public Data
{
public:
	static void Copy(Buffer *destination, const Buffer *source, size_t size, size_t destinationOffset = 0, size_t sourceOffset = 0);

	Buffer(size_t size) : m_buffer(size) {}
	Buffer(CUdeviceptr buffer, size_t size) : m_buffer(buffer, size) {}

	// Transfers

	void AllocateOnGPU() { m_buffer.AllocateOnGPU(); }
	void TransferToGPU() { m_buffer.TransferToGPU(); }

	void Clear(size_t offset = 0);
	void TransferToCPU();

	// Buffers

	const void *GetCPUBuffer() const { return m_CPUBuffer; }
	void SetCPUBuffer(void *buffer)
	{
		m_buffer.SetCPUBuffer(buffer);
		m_CPUBuffer = buffer;
	}
	bool HasCPUBuffer () const { return (m_CPUBuffer != nullptr); }

	CUdeviceptr GetGPUBuffer() const { return m_buffer.GetGPUBuffer(); }
	const void *GetAddress() const override { return m_buffer.GetAddress(); }

	bool IsAlias() const { return m_buffer.IsAlias(); }

	// Sizes

	size_t GetSize() const { return m_buffer.GetSize(); }
	void SetSize(size_t size) { m_buffer.SetSize(size); }

	size_t GetAllocatedSize() const { return m_buffer.GetAllocatedSize(); }

	// Tag

	const std::string& GetTag() const { return m_buffer.GetTag(); }
	void SetTag(const std::string &tag) { m_buffer.SetTag(tag); }

protected:
	ConstantBuffer m_buffer;
	void *m_CPUBuffer = nullptr;
};

}
