#pragma once

#include "Runtime/DataBuffers/DataBuffer.h"

#include "Analysis/Shape/Shape.h"

#include "HorseIR/Tree/Tree.h"

#include "Runtime/DataBuffers/ListBuffer.h"
#include "Runtime/DataBuffers/VectorBuffer.h"

#include "Utils/Logger.h"

namespace Runtime {

class DictionaryBuffer : public DataBuffer
{
public:
	constexpr static DataBuffer::Kind BufferKind = DataBuffer::Kind::Dictionary;

	DictionaryBuffer(VectorBuffer *keys, ListBuffer *values);
	~DictionaryBuffer() override;

	const HorseIR::DictionaryType *GetType() const override { return m_type; }
	const Analysis::DictionaryShape *GetShape() const override { return m_shape; }

	// Keys/values

	VectorBuffer *GetKeys() const { return m_keys; }
	ListBuffer *GetValues() const { return m_values; }

	// CPU/GPU management

	CUDA::Buffer *GetGPUWriteBuffer() override { Utils::Logger::LogError("Unable to allocate dictionary GPU buffer"); }
	CUDA::Buffer *GetGPUReadBuffer() const override { Utils::Logger::LogError("Unable to allocate dictionary GPU buffer"); }

	size_t GetGPUBufferSize() const override { return 0; }

	// Printers

	std::string Description() const override;
	std::string DebugDump() const override;

private:
	const HorseIR::DictionaryType *m_type = nullptr;
	const Analysis::DictionaryShape *m_shape = nullptr;

	VectorBuffer *m_keys = nullptr;
	ListBuffer *m_values = nullptr;

	unsigned int m_size = 0;
};

}

