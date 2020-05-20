#pragma once

#include <string>

#include "CUDA/Buffer.h"

#include "Analysis/Shape/Shape.h"

#include "HorseIR/Tree/Tree.h"

namespace Runtime {

class BufferUtils;
class DataBuffer
{
	friend class BufferUtils;
public:
	enum class Kind {
		Vector,
		List,
		Enumeration,
		Dictionary,
		Table,
		KeyedTable,
		Function,
		Constant
	};

	[[noreturn]] void CPUOnlyBuffer() const
	{
		Utils::Logger::LogError(KindString(m_kind) + " is a CPU only buffer");
	}

	static std::string KindString(Kind kind)
	{
		switch (kind)
		{
			case Kind::Vector:
				return "DataBuffer::Vector";
			case Kind::List:
				return "DataBuffer::List";
			case Kind::Enumeration:
				return "DataBuffer::Enumeration";
			case Kind::Dictionary:
				return "DataBuffer::Dictionary";
			case Kind::Table:
				return "DataBuffer::Table";
			case Kind::KeyedTable:
				return "DataBuffer::KeyedTable";
			case Kind::Function:
				return "DataBuffer::Function";
			case Kind::Constant:
				return "DataBuffer::Constant";
		}
		return "<unknown>";
	}

	static DataBuffer *CreateEmpty(const HorseIR::Type *type, const Analysis::Shape *shape);
	virtual ~DataBuffer() {}

	virtual DataBuffer *Clone() const = 0;

	// Type/Shape

	virtual const HorseIR::Type *GetType() const = 0;
	virtual const Analysis::Shape *GetShape() const = 0;

	// GPU/CPU management

	virtual void InvalidateCPU() { m_cpuConsistent = false; }
	virtual void InvalidateGPU() { m_gpuConsistent = false; }

	virtual void ValidateCPU(bool recursive = false) const
	{
		if (!m_cpuConsistent)
		{
			if (!IsAllocatedOnCPU())
			{
				AllocateCPUBuffer();
			}
			if (IsAllocatedOnGPU())
			{
				TransferToCPU();
			}
			m_cpuConsistent = true;
		}
	}

	virtual void ValidateGPU(bool recursive = false) const
	{
		if (!m_gpuConsistent)
		{
			if (!IsAllocatedOnGPU())
			{
				AllocateGPUBuffer();
			}
			if (IsAllocatedOnCPU())
			{
				TransferToGPU();
			}
			m_gpuConsistent = true;
		}
	}
	
	virtual CUDA::Data *GetGPUWriteBuffer() { CPUOnlyBuffer(); }
	virtual CUDA::Data *GetGPUReadBuffer() const { CPUOnlyBuffer(); }

	virtual CUDA::Buffer *GetGPUSizeBuffer() const { CPUOnlyBuffer(); }
	virtual size_t GetGPUBufferSize() const { CPUOnlyBuffer(); }

	virtual bool ReallocateGPUBuffer() { CPUOnlyBuffer(); }

	// Printers

	virtual std::string Description() const = 0;
	virtual std::string DebugDump() const = 0;

	// Clear

	enum class ClearMode {
		Zero,
		Minimum,
		Maximum
	};

	virtual void Clear(ClearMode mode = ClearMode::Zero) = 0;

protected:
	DataBuffer(Kind kind) : m_kind(kind) {}
	Kind m_kind;

	bool IsCPUConsistent() const { return m_cpuConsistent; }
	bool IsGPUConsistent() const { return m_gpuConsistent; }

	virtual bool IsAllocatedOnCPU() const { return true; }
	virtual bool IsAllocatedOnGPU() const { return false; }

	virtual void TransferToCPU() const { CPUOnlyBuffer(); }
	virtual void TransferToGPU() const { CPUOnlyBuffer(); }
	
	virtual void AllocateCPUBuffer() const { CPUOnlyBuffer(); }
	virtual void AllocateGPUBuffer() const { CPUOnlyBuffer(); }

	mutable bool m_gpuConsistent = false;
	mutable bool m_cpuConsistent = false;
};

}
