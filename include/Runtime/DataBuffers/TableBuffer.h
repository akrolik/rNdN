#pragma once

#include "Runtime/DataBuffers/DataBuffer.h"

#include <string>
#include <unordered_map>

#include "Analysis/Shape/Shape.h"

#include "HorseIR/Tree/Tree.h"

#include "Runtime/DataBuffers/VectorBuffer.h"

namespace Runtime {

class TableBuffer : public DataBuffer
{
public:
	constexpr static DataBuffer::Kind BufferKind = DataBuffer::Kind::Table;

	TableBuffer(const std::unordered_map<std::string, VectorBuffer *>& columns);

	HorseIR::TableType *GetType() const override { return m_type; }
	const Analysis::TableShape *GetShape() const override { return m_shape; }

	// Columns

	VectorBuffer *GetColumn(const std::string& column) const;

	// CPU/GPU management

	CUDA::Buffer *GetGPUWriteBuffer() override { Utils::Logger::LogError("Unable to allocate table GPU buffer"); }
	CUDA::Buffer *GetGPUReadBuffer() const override { Utils::Logger::LogError("Unable to allocate table GPU buffer"); }

	// Printers

	std::string Description() const override;
	std::string DebugDump() const override;

private:
	HorseIR::TableType *m_type = new HorseIR::TableType();
	Analysis::TableShape *m_shape = nullptr;

	std::unordered_map<std::string, VectorBuffer *> m_columns;
	unsigned int m_rows = 0;
};

}

