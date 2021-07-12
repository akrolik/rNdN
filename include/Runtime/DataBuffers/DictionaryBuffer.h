#pragma once

#include "Runtime/DataBuffers/DataBuffer.h"

#include "HorseIR/Analysis/Shape/Shape.h"
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

	// Tag

	void SetTag(const std::string& tag) override;

	// Type/shape

	const HorseIR::DictionaryType *GetType() const override { return m_type; }
	const HorseIR::Analysis::DictionaryShape *GetShape() const override { return m_shape; }

	// Keys/values

	const VectorBuffer *GetKeys() const { return m_keys; }
	VectorBuffer *GetKeys() { return m_keys; }

	const ListBuffer *GetValues() const { return m_values; }
	ListBuffer *GetValues() { return m_values; }

	// CPU/GPU management

	void ValidateCPU() const override;
	void ValidateGPU() const override;

	// Printers

	std::string Description() const override;
	std::string DebugDump(unsigned int indent = 0, bool preindent = false) const override;

	// Clear

	void Clear(ClearMode mode = ClearMode::Zero) override;

private:
	const HorseIR::DictionaryType *m_type = nullptr;
	const HorseIR::Analysis::DictionaryShape *m_shape = nullptr;

	VectorBuffer *m_keys = nullptr;
	ListBuffer *m_values = nullptr;

	unsigned int m_size = 0;
};

}

