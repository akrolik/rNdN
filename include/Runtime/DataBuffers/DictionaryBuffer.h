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

	DictionaryBuffer *Clone() const override;

	// Type/shape

	const HorseIR::DictionaryType *GetType() const override { return m_type; }
	const Analysis::DictionaryShape *GetShape() const override { return m_shape; }

	// Keys/values

	VectorBuffer *GetKeys() const { return m_keys; }
	ListBuffer *GetValues() const { return m_values; }

	// CPU/GPU management

	void ValidateCPU(bool recursive = false) const override;
	void ValidateGPU(bool recursive = false) const override;

	// Printers

	std::string Description() const override;
	std::string DebugDump() const override;

	// Clear

	void Clear(ClearMode mode = ClearMode::Zero) override;

private:
	const HorseIR::DictionaryType *m_type = nullptr;
	const Analysis::DictionaryShape *m_shape = nullptr;

	VectorBuffer *m_keys = nullptr;
	ListBuffer *m_values = nullptr;

	unsigned int m_size = 0;
};

}

