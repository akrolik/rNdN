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

	EnumerationBuffer *Clone() const override
	{
		return new EnumerationBuffer(m_keys->Clone(), m_values->Clone(), m_indexes->Clone());
	}

	// Type/shape

	const HorseIR::EnumerationType *GetType() const override { return m_type; }
	const Analysis::EnumerationShape *GetShape() const override { return m_shape; }

	// Keys/values

	VectorBuffer *GetKeys() const { return m_keys; }
	VectorBuffer *GetValues() const { return m_values; }
	TypedVectorBuffer<std::int64_t> *GetIndexes() const { return m_indexes; }

	unsigned int GetElementCount() const { return m_values->GetElementCount(); }

	// CPU/GPU management

	void ValidateCPU(bool recursive = false) const override
	{
		ColumnBuffer::ValidateCPU(recursive);
		if (recursive)
		{
			m_keys->ValidateCPU(true);
			m_values->ValidateCPU(true);
			m_indexes->ValidateCPU(true);
		}
	}

	void ValidateGPU(bool recursive = false) const override
	{
		ColumnBuffer::ValidateGPU(recursive);
		if (recursive)
		{
			m_keys->ValidateGPU(true);
			m_values->ValidateGPU(true);
			m_indexes->ValidateGPU(true);
		}
	}

	// Printers

	std::string Description() const override;
	std::string DebugDump() const override;
	std::string DebugDump(unsigned int index) const override;

	// Clear

	void Clear() override
	{
		m_keys->Clear();
		m_values->Clear();
	}

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

