#pragma once

#include "Runtime/DataBuffers/ColumnBuffer.h"

#include "HorseIR/Analysis/Shape/Shape.h"
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

	// Tag

	void SetTag(const std::string& tag) override;

	// Type/shape

	const HorseIR::EnumerationType *GetType() const override { return m_type; }
	const HorseIR::Analysis::EnumerationShape *GetShape() const override { return m_shape; }

	// Keys/values

	const VectorBuffer *GetKeys() const { return m_keys; }
	VectorBuffer *GetKeys() { return m_keys; }

	const VectorBuffer *GetValues() const { return m_values; }
	VectorBuffer *GetValues() { return m_values; }

	const TypedVectorBuffer<std::int64_t> *GetIndexes() const { return m_indexes; }
	TypedVectorBuffer<std::int64_t> *GetIndexes() { return m_indexes; }

	unsigned int GetElementCount() const { return m_values->GetElementCount(); }

	// CPU/GPU management

	void ValidateCPU() const override;
	void ValidateGPU() const override;

	// Printers

	std::string Description() const override;
	std::string DebugDump() const override;
	std::string DebugDump(unsigned int index) const override;

	// Clear

	void Clear(ClearMode mode = ClearMode::Zero) override;

private:
	const HorseIR::EnumerationType *m_type = nullptr;
	const HorseIR::Analysis::EnumerationShape *m_shape = nullptr;

	//TODO: Support multiple column enums
	VectorBuffer *m_keys = nullptr;
	VectorBuffer *m_values = nullptr;
	TypedVectorBuffer<std::int64_t> *m_indexes = nullptr;

	unsigned int m_size = 0;
};

}

