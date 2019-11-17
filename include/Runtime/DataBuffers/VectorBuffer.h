#pragma once

#include <typeindex>
#include <typeinfo>

#include "Runtime/DataBuffers/ColumnBuffer.h"
#include "Runtime/DataBuffers/DataObjects/VectorData.h"
#include "Runtime/StringBucket.h"

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

	virtual VectorData *GetCPUWriteBuffer() = 0;
	virtual const VectorData *GetCPUReadBuffer() const = 0;

	// Data size, useful for allocations

	unsigned int GetElementCount() const { return m_elementCount; }

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
		if constexpr(std::is_same<T, std::string>::value)
		{
			return sizeof(std::uint64_t) * m_elementCount;
		}
		else
		{
			return sizeof(T) * m_elementCount;
		}
	}

	std::string Description() const override
	{
		std::string string = HorseIR::PrettyPrinter::PrettyString(m_type) + "(";
		if constexpr(std::is_same<T, std::string>::value)
		{
			string += std::to_string(sizeof(std::uint64_t));
		}
		else
		{
			string += std::to_string(sizeof(T));
		}
		return string + " bytes) x " + std::to_string(m_elementCount);
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
	bool IsAllocatedOnCPU() const { return (m_cpuBuffer != nullptr); }
	bool IsAllocatedOnGPU() const { return (m_gpuBuffer != nullptr); }

	void AllocateCPUBuffer() const
	{
		m_cpuBuffer = new TypedVectorData<T>(m_type, m_elementCount);
	}

	void AllocateGPUBuffer() const
	{
		if constexpr(std::is_same<T, std::string>::value)
		{
			m_gpuBuffer = new CUDA::Buffer(sizeof(std::uint64_t) * m_elementCount);
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
			if (!IsAllocatedOnCPU())
			{
				AllocateCPUBuffer();
			}
			if (IsAllocatedOnGPU())
			{
				if constexpr(std::is_same<T, std::string>::value)
				{
					// Setup CPU buffer if needed

					if (!m_gpuBuffer->HasCPUBuffer())
					{
						auto hashedData = static_cast<std::uint64_t *>(malloc(sizeof(std::uint64_t) * m_elementCount));
						m_gpuBuffer->SetCPUBuffer(hashedData);
					}
					m_gpuBuffer->TransferToCPU();

					// Marshal data

					auto timeMarshalling_start = Utils::Chrono::Start();

					auto hashedData = static_cast<std::uint64_t *>(m_gpuBuffer->GetCPUBuffer());
					for (auto i = 0u; i < m_elementCount; ++i)
					{
						m_cpuBuffer->SetValue(i, StringBucket::RecoverString(hashedData[i]));
					}

					auto timeMarshalling = Utils::Chrono::End(timeMarshalling_start);
					Utils::Logger::LogTiming("Data marshalling (string)", timeMarshalling);
				}
				else
				{
					m_gpuBuffer->SetCPUBuffer(m_cpuBuffer->GetData());
					m_gpuBuffer->TransferToCPU();
				}
			}
			m_cpuConsistent = true;
		}
	}

	void ValidateGPU() const
	{
		if (!m_gpuConsistent)
		{
			if (!IsAllocatedOnGPU())
			{
				AllocateGPUBuffer();
			}
			if (IsAllocatedOnCPU())
			{
				if constexpr(std::is_same<T, std::string>::value)
				{
					if (!m_gpuBuffer->HasCPUBuffer())
					{
						auto hashedData = static_cast<std::uint64_t *>(malloc(sizeof(std::uint64_t) * m_elementCount));
						m_gpuBuffer->SetCPUBuffer(hashedData);
					}

					// Marshal data

					auto timeMarshalling_start = Utils::Chrono::Start();

					auto hashedData = static_cast<std::uint64_t *>(m_gpuBuffer->GetCPUBuffer());
					for (auto i = 0u; i < m_elementCount; ++i)
					{
						hashedData[i] = StringBucket::HashString(m_cpuBuffer->GetValue(i));
					}

					auto timeMarshalling = Utils::Chrono::End(timeMarshalling_start);
					Utils::Logger::LogTiming("Data marshalling (string)", timeMarshalling);

					m_gpuBuffer->TransferToGPU();
				}
				else
				{
					m_gpuBuffer->SetCPUBuffer(m_cpuBuffer->GetData());
					m_gpuBuffer->TransferToGPU();
				}
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
