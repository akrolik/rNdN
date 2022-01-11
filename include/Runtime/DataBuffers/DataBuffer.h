#pragma once

#include <string>

#include "CUDA/Buffer.h"

#include "HorseIR/Analysis/Shape/Shape.h"

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

	static DataBuffer *CreateEmpty(const HorseIR::Type *type, const HorseIR::Analysis::Shape *shape);
	virtual ~DataBuffer() {}

	virtual DataBuffer *Clone() const = 0;

	// Tag

	const std::string& GetTag() const { return m_tag; }
	virtual void SetTag(const std::string& tag) { m_tag = tag; }

	// Type/Shape

	virtual const HorseIR::Type *GetType() const = 0;
	virtual const HorseIR::Analysis::Shape *GetShape() const = 0;

	// CPU/GPU management

	virtual void RequireCPUConsistent(bool exclusive) const
	{
		if (!IsCPUConsistent())
		{
			if (!exclusive && !IsGPUConsistent())
			{
				Utils::Logger::LogError("Empty buffer cannot directly enter shared state" + m_tag);
			}

			auto timeStart = Utils::Chrono::Start(TransferString("CPU"));
			if (!IsAllocatedOnCPU())
			{
				AllocateCPUBuffer();
			}
			if (IsAllocatedOnGPU() && IsGPUConsistent())
			{
				TransferToCPU();
			}
			Utils::Chrono::End(timeStart);
		}
		SetCPUConsistent(exclusive);
	}

	virtual void RequireGPUConsistent(bool exclusive) const
	{
		if (!IsGPUConsistent())
		{
			if (!exclusive && !IsCPUConsistent())
			{
				Utils::Logger::LogError("Empty buffer cannot directly enter shared state" + m_tag);
			}

			auto timeStart = Utils::Chrono::Start(TransferString("GPU"));
			if (!IsAllocatedOnGPU())
			{
				AllocateGPUBuffer();
			}
			if (IsAllocatedOnCPU() && IsCPUConsistent())
			{
				TransferToGPU();
			}
			Utils::Chrono::End(timeStart);
		}
		SetGPUConsistent(exclusive);
	}

	virtual CUDA::Data *GetGPUWriteBuffer() { CPUOnlyBuffer(); }
	virtual const CUDA::Data *GetGPUReadBuffer() const { CPUOnlyBuffer(); }

	virtual const CUDA::Buffer *GetGPUSizeBuffer() const { CPUOnlyBuffer(); }
	virtual CUDA::Buffer *GetGPUSizeBuffer() { CPUOnlyBuffer(); }

	virtual size_t GetGPUBufferSize() const { CPUOnlyBuffer(); }

	virtual bool ReallocateGPUBuffer() { CPUOnlyBuffer(); }

	// Printers

	virtual std::string Description() const = 0;
	virtual std::string DebugDump(unsigned int indent = 0, bool preindent = false) const = 0;

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

	// GPU/CPU management

	void SetCPUConsistent(bool exclusive) const
	{
		m_cpuConsistent = true;
		m_gpuConsistent &= !exclusive;
	}

	void SetGPUConsistent(bool exclusive) const
	{
		m_gpuConsistent = true;
		m_cpuConsistent &= !exclusive;
	}

	bool IsCPUConsistent() const { return m_cpuConsistent; }
	bool IsGPUConsistent() const { return m_gpuConsistent; }

	virtual bool IsAllocatedOnCPU() const { return false; }
	virtual bool IsAllocatedOnGPU() const { return false; }

	virtual void TransferToCPU() const { CPUOnlyBuffer(); }
	virtual void TransferToGPU() const { CPUOnlyBuffer(); }
	
	virtual void AllocateCPUBuffer() const { CPUOnlyBuffer(); }
	virtual void AllocateGPUBuffer() const { CPUOnlyBuffer(); }

	mutable bool m_gpuConsistent = false;
	mutable bool m_cpuConsistent = false;

	std::string m_tag = "";
	std::string TransferString(const std::string& name) const
	{
		return "Transfer " + ((m_tag == "") ? "" : "'" + m_tag + "' ") + "to " + name;
	}
};

}
