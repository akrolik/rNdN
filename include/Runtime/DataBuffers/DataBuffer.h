#pragma once

#include <string>

#include "CUDA/Buffer.h"

#include "Analysis/Shape/Shape.h"

#include "HorseIR/Tree/Tree.h"

namespace Runtime {

class DataBuffer
{
public:
	static DataBuffer *Create(const HorseIR::Type *type, const Analysis::Shape *shape);

	virtual const HorseIR::Type *GetType() const = 0;

	// GPU/CPU management

	// virtual DataObject *GetCPUWriteBuffer() = 0;
	// virtual DataObject *GetCPUReadBuffer() = 0;

	virtual CUDA::Buffer *GetGPUWriteBuffer() = 0;
	virtual CUDA::Buffer *GetGPUReadBuffer() const = 0;

	bool IsGPUConsistent() const { return m_gpuConsistent; }
	bool IsCPUConsistent() const { return m_cpuConsistent; }

	// Printers

	virtual std::string Description() const = 0;
	virtual std::string DebugDump() const = 0;

protected:
	mutable bool m_gpuConsistent = false;
	mutable bool m_cpuConsistent = false;
};

}
