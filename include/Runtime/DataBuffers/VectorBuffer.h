#pragma once

#include <limits>
#include <typeindex>
#include <typeinfo>

#include "Runtime/DataBuffers/ColumnBuffer.h"
#include "Runtime/DataBuffers/DataObjects/VectorData.h"

#include "Analysis/Shape/Shape.h"

#include "HorseIR/Tree/Tree.h"

#include "Utils/Chrono.h"
#include "Utils/Logger.h"
#include "Utils/Options.h"

namespace Runtime {

class BufferUtils;
class VectorBuffer : public ColumnBuffer
{
	friend class BufferUtils;
public:
	constexpr static DataBuffer::Kind BufferKind = DataBuffer::Kind::Vector;

	static VectorBuffer *CreateEmpty(const HorseIR::BasicType *type, const Analysis::Shape::Size *size);
	~VectorBuffer() override;

	virtual VectorBuffer *Clone() const = 0;

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
	virtual bool Resize(unsigned int size) = 0;

	// Clear

	virtual void Clear(ClearMode mode = ClearMode::Zero) override = 0;

protected:
	VectorBuffer(const std::type_index &tid, const HorseIR::BasicType *type, unsigned long elementCount);

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
		if (m_cpuBuffer != nullptr)
		{
			auto timeDealloc_start = Utils::Chrono::Start("CPU free (" + std::to_string(m_elementCount * sizeof(T)) + " bytes)");
			delete m_cpuBuffer;
			Utils::Chrono::End(timeDealloc_start);
		}
		delete m_gpuBuffer;
	}

	TypedVectorBuffer<T> *Clone() const override
	{
		// Initialize the vector buffer either with the CPU contents, or an empty buffer that will presumably hold GPU data

		TypedVectorBuffer<T> *clone = nullptr;
		if (IsCPUConsistent())
		{
			clone = new TypedVectorBuffer<T>(m_cpuBuffer);
		}
		else
		{
			clone = new TypedVectorBuffer<T>(m_type, m_elementCount);
		}

		// Copy GPU contents if consistent. This will also allocate the size

		if (IsGPUConsistent())
		{
			CUDA::Buffer::Copy(clone->GetGPUWriteBuffer(), GetGPUReadBuffer(), GetGPUBufferSize());
		}
		return clone;
	}

	// Sizing

	bool Resize(unsigned int size) override
	{
		// Update the container size and shape if necessary

		if (m_elementCount != size)
		{
			auto oldDescription = Description();

			// CPU buffer resize, does not need to be consistent, only allocated

			if (IsAllocatedOnCPU())
			{
				m_cpuBuffer->Resize(size);
			}

			// GPU buffer resize

			if (IsAllocatedOnGPU())
			{
				// Allocate a new buffer and transfer the contents if substantially smaller than the allocated size

				auto newBufferSize = sizeof(T) * size;
				if (newBufferSize < m_gpuBuffer->GetAllocatedSize() * 0.9)
				{
					// Copy active data range if needed, otherwise just reallocate

					auto oldBuffer = m_gpuBuffer;

					m_gpuBuffer = new CUDA::Buffer(newBufferSize);
					m_gpuBuffer->AllocateOnGPU();

					if (IsGPUConsistent())
					{
						CUDA::Buffer::Copy(m_gpuBuffer, oldBuffer, newBufferSize);
					}

					delete oldBuffer;
				}
				else
				{
					m_gpuBuffer->SetSize(newBufferSize);
				}

				// Update GPU size buffer if necessary

				if (m_gpuSize != size)
				{
					m_gpuSize = size;
					m_gpuSizeBuffer->TransferToGPU();
				}
			}

			m_elementCount = size;

			delete m_shape;
			m_shape = new Analysis::VectorShape(new Analysis::Shape::ConstantSize(m_elementCount));

			if (Utils::Options::Present(Utils::Options::Opt_Print_debug))
			{
				Utils::Logger::LogDebug("Resized vector buffer [" + oldDescription + "] to [" + Description() + "]");
			}

			return true;
		}
		return false;
	}

	// GPU functions

	bool IsAllocatedOnCPU() const override { return (m_cpuBuffer != nullptr); }
	bool IsAllocatedOnGPU() const override { return (m_gpuBuffer != nullptr); }

	TypedVectorData<T> *GetCPUWriteBuffer() override
	{
		ValidateCPU();
		InvalidateGPU();
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
		InvalidateCPU();
		return m_gpuBuffer;
	}

	CUDA::Buffer *GetGPUReadBuffer() const override
	{
		ValidateGPU();
		return m_gpuBuffer;
	}

	size_t GetGPUBufferSize() const override
	{
		return (sizeof(T) * m_gpuSize);
	}

	CUDA::Buffer *GetGPUSizeBuffer() const override
	{
		return m_gpuSizeBuffer;
	}

	bool ReallocateGPUBuffer() override
	{
		// Only resize from the GPU data size if the data is GPU resident

		if (IsGPUConsistent() && IsAllocatedOnGPU())
		{
			m_gpuSizeBuffer->TransferToCPU();
			return Resize(m_gpuSize);
		}
		return false;
	}

	// Printing utils

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

	void Clear(ClearMode mode = ClearMode::Zero) override
	{
		if (mode == ClearMode::Zero)
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
		else
		{
			auto data = GetCPUWriteBuffer();
			auto val = (mode == ClearMode::Maximum) ? std::numeric_limits<T>::max() : std::numeric_limits<T>::min();
			for (auto i = 0u; i < m_elementCount; ++i)
			{
				data->SetValue(i, val);
			}
		}
	}

protected:

	void AllocateCPUBuffer() const override
	{
		auto timeAlloc_start = Utils::Chrono::Start("CPU allocation (" + std::to_string(m_elementCount * sizeof(T)) + " bytes)");
		m_cpuBuffer = new TypedVectorData<T>(m_type, m_elementCount);
		Utils::Chrono::End(timeAlloc_start);
	}

	void AllocateGPUBuffer() const override
	{
		m_gpuBuffer = new CUDA::Buffer(sizeof(T) * m_elementCount);
		m_gpuBuffer->AllocateOnGPU();

		m_gpuSize = m_elementCount;
		m_gpuSizeBuffer = new CUDA::Buffer(&m_gpuSize, sizeof(std::uint32_t));
		m_gpuSizeBuffer->AllocateOnGPU();
		m_gpuSizeBuffer->TransferToGPU();
	}

	void TransferToCPU() const override
	{
		m_gpuBuffer->SetCPUBuffer(m_cpuBuffer->GetData());
		m_gpuBuffer->TransferToCPU();

		m_gpuSizeBuffer->TransferToCPU();
	}

	void TransferToGPU() const override
	{
		m_gpuBuffer->SetCPUBuffer(m_cpuBuffer->GetData());
		m_gpuBuffer->TransferToGPU();

		m_gpuSizeBuffer->TransferToGPU();
	}

	mutable TypedVectorData<T> *m_cpuBuffer = nullptr;
	mutable CUDA::Buffer *m_gpuBuffer = nullptr;

	// GPU size

	mutable CUDA::Buffer *m_gpuSizeBuffer = nullptr;
	mutable std::uint32_t m_gpuSize = 0;
};

}
