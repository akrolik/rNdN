#pragma once

#include <typeindex>
#include <typeinfo>

#include "Runtime/DataBuffers/ColumnBuffer.h"
#include "Runtime/DataBuffers/DataObjects/VectorData.h"
#include "Runtime/StringBucket.h"

#include "Analysis/Shape/Shape.h"

#include "HorseIR/Tree/Tree.h"

namespace Runtime {

class BufferUtils;
class VectorBuffer : public ColumnBuffer
{
	friend class BufferUtils;
public:
	constexpr static DataBuffer::Kind BufferKind = DataBuffer::Kind::Vector;

	static VectorBuffer *Create(const HorseIR::BasicType *type, const Analysis::Shape::Size *size);
	~VectorBuffer() override;

	// Type/Shape

	const HorseIR::BasicType *GetType() const override { return m_type; }
	const Analysis::VectorShape *GetShape() const override { return m_shape; }

	// GPU/CPU buffer management

	virtual VectorData *GetCPUWriteBuffer() = 0;
	virtual VectorData *GetCPUReadBuffer() const = 0;

	// Data size, useful for allocations

	unsigned int GetElementCount() const { return m_elementCount; }
	size_t GetElementSize() const { return m_elementSize; }

protected:
	VectorBuffer(const std::type_index &tid, const HorseIR::BasicType *type, unsigned long elementCount, size_t elementSize) :
		ColumnBuffer(DataBuffer::Kind::Vector), m_typeid(tid), m_elementCount(elementCount), m_elementSize(elementSize)
	{
		m_type = type->Clone();
		m_shape = new Analysis::VectorShape(new Analysis::Shape::ConstantSize(m_elementCount));
	}

	std::type_index m_typeid;

	const HorseIR::BasicType *m_type = nullptr;
	const Analysis::VectorShape *m_shape = nullptr;

	unsigned long m_elementCount = 0;
	size_t m_elementSize = 0;
};

template<typename T>
class TypedVectorBuffer : public VectorBuffer
{
public:
	TypedVectorBuffer(const HorseIR::BasicType *elementType, unsigned long elementCount) : VectorBuffer(typeid(T), elementType, elementCount, sizeof(T))
	{
		//TODO: Inefficient
		if constexpr(std::is_same<T, std::string>::value)
		{
			m_elementSize = sizeof(std::uint64_t);
		}
	}

	TypedVectorBuffer(TypedVectorData<T> *buffer) : VectorBuffer(typeid(T), buffer->GetType(), buffer->GetElementCount(), sizeof(T)), m_cpuBuffer(buffer)
	{
		if constexpr(std::is_same<T, std::string>::value)
		{
			m_elementSize = sizeof(std::uint64_t);
		}
	}

	~TypedVectorBuffer() override
	{
		delete m_cpuBuffer;
		delete m_gpuBuffer;
	}

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

	std::string DebugDump(unsigned int index) const override
	{
		return GetCPUReadBuffer()->DebugDump(index);
	}

private:
	void AllocateCPUBuffer() const
	{
		m_cpuBuffer = new TypedVectorData<T>(m_type, m_elementCount);
		if (m_gpuBuffer != nullptr)
		{
			if constexpr(std::is_same<T, std::string>::value)
			{
				std::uint64_t *hashedData = (std::uint64_t *)malloc(sizeof(std::uint64_t) * m_elementCount);
				m_gpuBuffer->SetCPUBuffer(hashedData);
			}
			else
			{
				m_gpuBuffer->SetCPUBuffer(m_cpuBuffer->GetData());
			}
		}
	}

	void AllocateGPUBuffer() const
	{
		if (m_cpuBuffer != nullptr)
		{
			if constexpr(std::is_same<T, std::string>::value)
			{
				std::uint64_t *hashedData = (std::uint64_t *)malloc(sizeof(std::uint64_t) * m_elementCount);
				for (auto i = 0u; i < m_elementCount; ++i)
				{
					hashedData[i] = StringBucket::HashString(m_cpuBuffer->GetValue(i));
				}
				m_gpuBuffer = new CUDA::Buffer(hashedData, sizeof(std::uint64_t) * m_elementCount);
			}
			else
			{
				m_gpuBuffer = new CUDA::Buffer(m_cpuBuffer->GetData(), sizeof(T) * m_elementCount);
			}
		}
		else
		{
			if constexpr(std::is_same<T, std::string>::value)
			{
				m_gpuBuffer = new CUDA::Buffer(sizeof(std::uint64_t) * m_elementCount);
			}
			else
			{
				m_gpuBuffer = new CUDA::Buffer(sizeof(T) * m_elementCount);
			}
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
				if constexpr(std::is_same<T, std::string>::value)
				{
					std::uint64_t *hashedData = (std::uint64_t *)m_gpuBuffer->GetCPUBuffer();
					for (auto i = 0u; i < m_elementCount; ++i)
					{
						m_cpuBuffer->SetValue(i, StringBucket::RecoverString(hashedData[i]));
					}
				}
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

	mutable TypedVectorData<T> *m_cpuBuffer = nullptr;
	mutable CUDA::Buffer *m_gpuBuffer = nullptr;
};

}
