#pragma once

#include "Runtime/DataBuffers/ColumnBuffer.h"

#include "Analysis/Shape/Shape.h"

#include "HorseIR/Tree/Tree.h"

#include "Runtime/DataBuffers/VectorBuffer.h"

namespace Runtime {

class EnumerationBuffer : public ColumnBuffer
{
public:
	constexpr static DataBuffer::Kind BufferKind = DataBuffer::Kind::Enumeration;

	EnumerationBuffer(VectorBuffer *keys, VectorBuffer *values);
	~EnumerationBuffer() override;

	const HorseIR::EnumerationType *GetType() const override { return m_type; }
	const Analysis::EnumerationShape *GetShape() const override { return m_shape; }

	// Keys/values

	VectorBuffer *GetKeys() const { return m_keys; }
	VectorBuffer *GetValues() const { return m_values; }

	unsigned int GetElementCount() const { return m_values->GetElementCount(); }

	// CPU/GPU management

	CUDA::Buffer *GetGPUWriteBuffer() override { Utils::Logger::LogError("Unable to allocate enumeration GPU buffer"); }
	CUDA::Buffer *GetGPUReadBuffer() const override { Utils::Logger::LogError("Unable to allocate enumeration GPU buffer"); }

	size_t GetGPUBufferSize() const override { return 0; }

	// Printers

	std::string Description() const override;
	std::string DebugDump() const override;
	std::string DebugDump(unsigned int index) const override;

private:
	const HorseIR::EnumerationType *m_type = nullptr;
	const Analysis::EnumerationShape *m_shape = nullptr;

	//TODO: Support multiple column enums
	VectorBuffer *m_keys = nullptr;
	VectorBuffer *m_values = nullptr;

	unsigned int m_size = 0;
};

}

