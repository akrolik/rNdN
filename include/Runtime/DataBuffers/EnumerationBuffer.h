#pragma once

#include "Runtime/DataBuffers/ColumnBuffer.h"

#include "Analysis/Shape/Shape.h"

#include "HorseIR/Tree/Tree.h"

#include "Runtime/DataBuffers/VectorBuffer.h"

#include "Utils/Logger.h"

namespace Runtime {

class EnumerationBuffer : public ColumnBuffer
{
public:
	constexpr static DataBuffer::Kind BufferKind = DataBuffer::Kind::Enumeration;

	EnumerationBuffer(VectorBuffer *keys, VectorBuffer *values, TypedVectorBuffer<std::int64_t> *indexes);
	~EnumerationBuffer() override;

	EnumerationBuffer *Clone() const override;

	// Type/shape

	const HorseIR::EnumerationType *GetType() const override { return m_type; }
	const Analysis::EnumerationShape *GetShape() const override { return m_shape; }

	// Keys/values

	VectorBuffer *GetKeys() const { return m_keys; }
	VectorBuffer *GetValues() const { return m_values; }
	TypedVectorBuffer<std::int64_t> *GetIndexes() const { return m_indexes; }

	unsigned int GetElementCount() const { return m_values->GetElementCount(); }

	// CPU/GPU management

	void ValidateCPU(bool recursive = false) const override;
	void ValidateGPU(bool recursive = false) const override;

	// Printers

	std::string Description() const override;
	std::string DebugDump() const override;
	std::string DebugDump(unsigned int index) const override;

	// Clear

	void Clear(ClearMode mode = ClearMode::Zero) override;

private:
	const HorseIR::EnumerationType *m_type = nullptr;
	const Analysis::EnumerationShape *m_shape = nullptr;

	//TODO: Support multiple column enums
	VectorBuffer *m_keys = nullptr;
	VectorBuffer *m_values = nullptr;
	TypedVectorBuffer<std::int64_t> *m_indexes = nullptr;

	unsigned int m_size = 0;
};

}

