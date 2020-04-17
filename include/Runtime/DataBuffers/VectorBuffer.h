#pragma once

#include <typeindex>
#include <typeinfo>

#include "Runtime/DataBuffers/ColumnBuffer.h"
#include "Runtime/DataBuffers/DataObjects/VectorData.h"

#include "Analysis/Shape/Shape.h"

#include "HorseIR/Tree/Tree.h"

#include "Utils/Chrono.h"

namespace Runtime {

class BufferUtils;
class VectorBuffer : public ColumnBuffer
{
	friend class BufferUtils;
public:
	constexpr static DataBuffer::Kind BufferKind = DataBuffer::Kind::Vector;

	static VectorBuffer *CreateEmpty(const HorseIR::BasicType *type, const Analysis::Shape::Size *size);
	~VectorBuffer() override;

	// Type/Shape

	const HorseIR::BasicType *GetType() const override { return m_type; }
	const Analysis::VectorShape *GetShape() const override { return m_shape; }

	// GPU/CPU buffer management

	virtual CUDA::Buffer *GetGPUWriteBuffer() override = 0;
	virtual CUDA::Buffer *GetGPUReadBuffer() const override = 0;

	virtual VectorData *GetCPUWriteBuffer() = 0;
	virtual const VectorData *GetCPUReadBuffer() const = 0;

	// Data size, useful for allocations

	unsigned int GetElementCount() const { return m_elementCount; }

	// Clear

	virtual void Clear() override = 0;

protected:
	VectorBuffer(const std::type_index &tid, const HorseIR::BasicType *type, unsigned long elementCount) :
		ColumnBuffer(DataBuffer::Kind::Vector), m_typeid(tid), m_elementCount(elementCount)
	{
		m_type = type->Clone();
		m_shape = new Analysis::VectorShape(new Analysis::Shape::ConstantSize(m_elementCount));
	}

	std::type_index m_typeid;

	const HorseIR::BasicType *m_type = nullptr;
	const Analysis::VectorShape *m_shape = nullptr;

	unsigned long m_elementCount = 0;
};

template<typename T>
class TypedVectorBuffer : public VectorBuffer
{
public:
	TypedVectorBuffer(const HorseIR::BasicType *elementType, unsigned long elementCount) : VectorBuffer(typeid(T), elementType, elementCount) {}
	TypedVectorBuffer(TypedVectorData<T> *buffer) : VectorBuffer(typeid(T), buffer->GetType(), buffer->GetElementCount()), m_cpuBuffer(buffer) {}

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

	bool IsAllocatedOnCPU() const override { return (m_cpuBuffer != nullptr); }
	bool IsAllocatedOnGPU() const override { return (m_gpuBuffer != nullptr); }

	const TypedVectorData<T> *GetCPUReadBuffer() const override
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

	size_t GetGPUBufferSize() const override
	{
		return sizeof(T) * m_elementCount;
	}

	std::string Description() const override
	{
		std::string string;
		string += HorseIR::PrettyPrinter::PrettyString(m_type) + "(";
		string += std::to_string(sizeof(T));
		string += " bytes) x " + std::to_string(m_elementCount);
		return string;
	}

	std::string DebugDump() const override
	{
		return GetCPUReadBuffer()->DebugDump();
	}

	std::string DebugDump(unsigned int index) const override
	{
		return GetCPUReadBuffer()->DebugDump(index);
	}

	// Clear

	void Clear() override
	{
		if (IsCPUConsistent())
		{
			GetCPUWriteBuffer()->Clear();
		}
		if (IsGPUConsistent())
		{
			GetGPUWriteBuffer()->Clear();
		}
	}

protected:

	void AllocateCPUBuffer() const override
	{
		m_cpuBuffer = new TypedVectorData<T>(m_type, m_elementCount);
	}

	void AllocateGPUBuffer() const override
	{
		m_gpuBuffer = new CUDA::Buffer(sizeof(T) * m_elementCount);
		m_gpuBuffer->AllocateOnGPU();
	}

	void TransferToCPU() const override
	{
		m_gpuBuffer->SetCPUBuffer(m_cpuBuffer->GetData());
		m_gpuBuffer->TransferToCPU();
	}

	void TransferToGPU() const override
	{
		m_gpuBuffer->SetCPUBuffer(m_cpuBuffer->GetData());
		m_gpuBuffer->TransferToGPU();
	}

	mutable TypedVectorData<T> *m_cpuBuffer = nullptr;
	mutable CUDA::Buffer *m_gpuBuffer = nullptr;
};

}
