#pragma once

#include "Runtime/DataBuffers/DataBuffer.h"

#include "Analysis/Shape/Shape.h"

#include "HorseIR/Tree/Tree.h"

#include "Runtime/DataBuffers/DataObjects/VectorData.h"

namespace Runtime {

class VectorBuffer : public DataBuffer
{
public:
	static VectorBuffer *Create(const HorseIR::BasicType *type, const Analysis::VectorShape *shape);

	virtual VectorData *GetCPUWriteBuffer() = 0;
	virtual VectorData *GetCPUReadBuffer() const = 0;

	virtual size_t GetElementCount() const = 0;
	virtual size_t GetElementSize() const = 0;
};

template<typename T>
class TypedVectorBuffer : public VectorBuffer
{
public:
	TypedVectorBuffer(const HorseIR::BasicType *elementType, unsigned long elementCount) : m_elementType(elementType), m_elementCount(elementCount) {}

	TypedVectorBuffer(TypedVectorData<T> *buffer) : m_elementType(buffer->GetType()), m_elementCount(buffer->GetElementCount()), m_cpuBuffer(buffer) {}

	const HorseIR::BasicType *GetType() const { return m_elementType; }

	size_t GetElementCount() const override { return m_elementCount; }
	size_t GetElementSize() const override { return sizeof(T); }

	TypedVectorData<T> *GetCPUWriteBuffer() override
	{
		ValidateCPU();
		m_gpuConsistent = false;
		return m_cpuBuffer;
	}

	TypedVectorData<T> *GetCPUReadBuffer() const override
	{
		ValidateCPU();
		return m_cpuBuffer;
	}

	CUDA::Buffer *GetGPUWriteBuffer() override
	{
		ValidateGPU();
		m_cpuConsistent = false;
		return m_gpuBuffer;
	}

	CUDA::Buffer *GetGPUReadBuffer() const override
	{
		ValidateGPU();
		return m_gpuBuffer;
	}

	std::string Description() const override
	{
		return GetCPUReadBuffer()->Description();
	}

	std::string DebugDump() const override
	{
		return GetCPUReadBuffer()->DebugDump();
	}

private:
	void AllocateCPUBuffer() const
	{
		m_cpuBuffer = new TypedVectorData<T>(m_elementType, m_elementCount);
		if (m_gpuBuffer != nullptr)
		{
			m_gpuBuffer->SetCPUBuffer(m_cpuBuffer->GetData());
		}
	}

	void AllocateGPUBuffer() const
	{
		m_gpuBuffer = new CUDA::Buffer(m_cpuBuffer->GetData(), sizeof(T) * m_elementCount);
		m_gpuBuffer->AllocateOnGPU();
	}

	void ValidateCPU() const
	{
		if (!m_cpuConsistent)
		{
			if (m_cpuBuffer == nullptr)
			{
				AllocateCPUBuffer();
			}
			if (m_gpuBuffer != nullptr)
			{
				m_gpuBuffer->TransferToCPU();
			}
			m_cpuConsistent = true;
		}
	}

	void ValidateGPU() const
	{
		if (!m_gpuConsistent)
		{
			if (m_gpuBuffer == nullptr)
			{
				AllocateGPUBuffer();
			}
			if (m_cpuBuffer != nullptr)
			{
				m_gpuBuffer->TransferToGPU();
			}
			m_gpuConsistent = true;
		}
	}

	const HorseIR::BasicType *m_elementType = nullptr;
	unsigned long m_elementCount = 0;

	mutable TypedVectorData<T> *m_cpuBuffer = nullptr;
	mutable CUDA::Buffer *m_gpuBuffer = nullptr;
};

}
