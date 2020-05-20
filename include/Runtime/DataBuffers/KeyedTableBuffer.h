#pragma once

#include "Runtime/DataBuffers/DataBuffer.h"

#include "Analysis/Shape/Shape.h"

#include "HorseIR/Tree/Tree.h"

#include "Runtime/DataBuffers/TableBuffer.h"

#include "Utils/Logger.h"

namespace Runtime {

class KeyedTableBuffer : public DataBuffer
{
public:
	constexpr static DataBuffer::Kind BufferKind = DataBuffer::Kind::KeyedTable;

	KeyedTableBuffer(TableBuffer *key, TableBuffer *value);
	~KeyedTableBuffer() override;

	KeyedTableBuffer *Clone() const override
	{
		return new KeyedTableBuffer(m_key->Clone(), m_value->Clone());
	}

	// Type/shape

	const HorseIR::KeyedTableType *GetType() const override { return m_type; }
	const Analysis::KeyedTableShape *GetShape() const override { return m_shape; }

	// Keys/values

	TableBuffer *GetKey() const { return m_key; }
	TableBuffer *GetValue() const { return m_value; }

	// CPU/GPU management

	void ValidateCPU(bool recursive = false) const override
	{
		DataBuffer::ValidateCPU(recursive);
		if (recursive)
		{
			m_key->ValidateCPU(true);
			m_value->ValidateCPU(true);
		}
	}

	void ValidateGPU(bool recursive = false) const override
	{
		DataBuffer::ValidateGPU(recursive);
		if (recursive)
		{
			m_key->ValidateGPU(true);
			m_value->ValidateGPU(true);
		}
	}

	// Printers

	std::string Description() const override;
	std::string DebugDump() const override;

	// Clear

	void Clear(ClearMode mode = ClearMode::Zero) override
	{
		if (mode == ClearMode::Zero)
		{
			m_key->Clear(mode);
			m_value->Clear(mode);
		}
	}

private:
	const HorseIR::KeyedTableType *m_type = nullptr;
	const Analysis::KeyedTableShape *m_shape = nullptr;

	TableBuffer *m_key = nullptr;
	TableBuffer *m_value = nullptr;
};

}

