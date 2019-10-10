#pragma once

#include <typeindex>
#include <typeinfo>

#include "Runtime/DataBuffers/DataBuffer.h"

#include "Analysis/Shape/Shape.h"

#include "HorseIR/Tree/Tree.h"

#include "Runtime/DataBuffers/DataObjects/VectorData.h"

namespace Runtime {

class BufferUtils;
class VectorBuffer : public DataBuffer
{
	friend class BufferUtils;
public:
	constexpr static DataBuffer::Kind BufferKind = DataBuffer::Kind::Vector;

	static VectorBuffer *Create(const HorseIR::BasicType *type, const Analysis::VectorShape *shape);

	virtual VectorData *GetCPUWriteBuffer() = 0;
	virtual VectorData *GetCPUReadBuffer() const = 0;

	virtual size_t GetElementCount() const = 0;
	virtual size_t GetElementSize() const = 0;

protected:
	VectorBuffer(const std::type_index &tid) : DataBuffer(DataBuffer::Kind::Vector), m_typeid(tid) {}

	std::type_index m_typeid;
};

template<typename T>
class TypedVectorBuffer : public VectorBuffer
{
public:
	TypedVectorBuffer(const HorseIR::BasicType *elementType, unsigned long elementCount) : VectorBuffer(typeid(T)), m_type(elementType), m_elementCount(elementCount)
	{
		m_shape = new Analysis::VectorShape(new Analysis::Shape::ConstantSize(m_elementCount));
	}

	TypedVectorBuffer(TypedVectorData<T> *buffer) : VectorBuffer(typeid(T)), m_type(buffer->GetType()), m_elementCount(buffer->GetElementCount()), m_cpuBuffer(buffer)
	{
		m_shape = new Analysis::VectorShape(new Analysis::Shape::ConstantSize(m_elementCount));
	}

	const HorseIR::BasicType *GetType() const override { return m_type; }
	const Analysis::VectorShape *GetShape() const override { return m_shape; }

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
		return (HorseIR::PrettyPrinter::PrettyString(m_type) + "(" + std::to_string(GetElementSize()) + " bytes) x " + std::to_string(GetElementCount()));
	}

	std::string DebugDump() const override
	{
		return GetCPUReadBuffer()->DebugDump();
	}

private:
	void AllocateCPUBuffer() const
	{
		m_cpuBuffer = new TypedVectorData<T>(m_type, m_elementCount);
		if (m_gpuBuffer != nullptr)
		{
			m_gpuBuffer->SetCPUBuffer(m_cpuBuffer->GetData());
		}
	}

	void AllocateGPUBuffer() const
	{
		if (m_cpuBuffer != nullptr)
		{
			m_gpuBuffer = new CUDA::Buffer(m_cpuBuffer->GetData(), sizeof(T) * m_elementCount);
		}
		else
		{
			m_gpuBuffer = new CUDA::Buffer(sizeof(T) * m_elementCount);
		}
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
			else
			{
				m_gpuBuffer->Clear();
			}
			m_gpuConsistent = true;
		}
	}

	const HorseIR::BasicType *m_type = nullptr;
	const Analysis::VectorShape *m_shape = nullptr;
	unsigned long m_elementCount = 0;

	mutable TypedVectorData<T> *m_cpuBuffer = nullptr;
	mutable CUDA::Buffer *m_gpuBuffer = nullptr;
};

}
