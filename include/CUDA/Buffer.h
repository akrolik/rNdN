#pragma once

#include <cuda.h>
#include <cuda_runtime.h>
#include <string>

#include "CUDA/Data.h"
 
namespace CUDA {

class ConstantBuffer : public Data
{
public:
	ConstantBuffer(size_t size) : ConstantBuffer(nullptr, size) {}
	ConstantBuffer(const void *buffer, size_t size);

	~ConstantBuffer();

	// Transfers

	void AllocateOnGPU();
	void Clear();
	void TransferToGPU();

	// Buffers

	virtual const void *GetCPUBuffer() const { return m_CPUBuffer; }
	virtual void SetCPUBuffer(const void *buffer) { m_CPUBuffer = buffer; }
	virtual bool HasCPUBuffer () const { return (m_CPUBuffer != nullptr); }

	CUdeviceptr& GetGPUBuffer() { return m_GPUBuffer; }
	void *GetAddress() override { return &m_GPUBuffer; }

	// Sizes

	size_t GetSize() const { return m_size; }
	void SetSize(size_t size) { m_size = size; }

	size_t GetAllocatedSize() const { return m_allocatedSize; }

	// Tag

	const std::string& GetTag() const { return m_tag; }
	void SetTag(const std::string &tag) { m_tag = tag; }

protected:
	std::string ChronoDescription(const std::string& operation, size_t size);

	const void *m_CPUBuffer = nullptr;
	CUdeviceptr m_GPUBuffer;

	size_t m_size = 0;
	size_t m_allocatedSize = 0;

	std::string m_tag;
};

class Buffer : public ConstantBuffer
{
public:
	static void Copy(Buffer *destination, ConstantBuffer *source, size_t size, size_t destinationOffset = 0, size_t sourceOffset = 0);

	Buffer(size_t size) : Buffer(nullptr, size) {}
	Buffer(void *buffer, size_t size) : ConstantBuffer(buffer, size), m_CPUBuffer(buffer) {}

	// Transfers

	void TransferToCPU();

	// Buffers

	const void *GetCPUBuffer() const override { return m_CPUBuffer; }

	void SetCPUBuffer(const void *buffer) override { ConstantBuffer::SetCPUBuffer(buffer); }
	void SetCPUBuffer(void *buffer)
	{
		SetCPUBuffer(static_cast<const void *>(buffer));
		m_CPUBuffer = buffer;
	}

	bool HasCPUBuffer () const override { return (m_CPUBuffer != nullptr); }

protected:
	void *m_CPUBuffer = nullptr;
};

}
