#pragma once

#include "Runtime/DataBuffers/DataBuffer.h"

#include "HorseIR/Analysis/Shape/Shape.h"
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

	KeyedTableBuffer *Clone() const override;

	// Tag

	void SetTag(const std::string& tag) override;

	// Type/shape

	const HorseIR::KeyedTableType *GetType() const override { return m_type; }
	const HorseIR::Analysis::KeyedTableShape *GetShape() const override { return m_shape; }

	// Keys/values

	const TableBuffer *GetKey() const { return m_key; }
	TableBuffer *GetKey() { return m_key; }

	const TableBuffer *GetValue() const { return m_value; }
	TableBuffer *GetValue() { return m_value; }

	// CPU/GPU management

	void RequireCPUConsistent(bool exclusive) const override;
	void RequireGPUConsistent(bool exclusive) const override;

	// Printers

	std::string Description() const override;
	std::string DebugDump(unsigned int indent = 0, bool preindent = false) const override;

	// Clear

	void Clear(ClearMode mode = ClearMode::Zero) override;

private:
	// Type/shape

	const HorseIR::KeyedTableType *m_type = nullptr;
	const HorseIR::Analysis::KeyedTableShape *m_shape = nullptr;

	// Data

	TableBuffer *m_key = nullptr;
	TableBuffer *m_value = nullptr;
};

}

