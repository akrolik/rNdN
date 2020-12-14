#pragma once

#include <cuda.h>
#include <cuda_runtime.h>
#include <string>

#include "CUDA/Data.h"

namespace CUDA {

class ConstantBuffer : public Data
{
public:
	ConstantBuffer(size_t size);
	ConstantBuffer(CUdeviceptr buffer, size_t size);

	~ConstantBuffer();

	// Transfers

	void AllocateOnGPU();
	void Clear();
	void TransferToGPU();

	// Buffers

	const void *GetCPUBuffer() const { return m_CPUBuffer; }
	void SetCPUBuffer(const void *buffer) { m_CPUBuffer = buffer; }
	bool HasCPUBuffer () const { return (m_CPUBuffer != nullptr); }

	CUdeviceptr GetGPUBuffer() const { return m_GPUBuffer; }
	const void *GetAddress() const override { return &m_GPUBuffer; }

	bool IsAlias() const { return m_alias; }

	// Sizes

	size_t GetSize() const { return m_size; }
	void SetSize(size_t size) { m_size = size; }

	size_t GetAllocatedSize() const { return m_allocatedSize; }

	// Tag

	const std::string& GetTag() const { return m_tag; }
	void SetTag(const std::string &tag) { m_tag = tag; }

	std::string ChronoDescription(const std::string& operation, size_t size);

protected:
	const void *m_CPUBuffer = nullptr;
	CUdeviceptr m_GPUBuffer = 0;

	size_t m_size = 0;
	size_t m_allocatedSize = 0;

	std::string m_tag;
	bool m_alias = false;
};

}
