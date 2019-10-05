#pragma once

#include "Runtime/DataBuffers/DataBuffer.h"

#include <string>
#include <unordered_map>

#include "HorseIR/Tree/Tree.h"

#include "Runtime/DataBuffers/VectorBuffer.h"

namespace Runtime {

class TableBuffer : public DataBuffer
{
public:
	TableBuffer(unsigned long rows) : m_rows(rows) {}

	HorseIR::TableType *GetType() const { return m_type; }

	// Columns

	void AddColumn(const std::string& name, VectorBuffer *column);
	VectorBuffer *GetColumn(const std::string& column) const;

	unsigned long GetRows() const { return m_rows; }

	// CPU/GPU management

	CUDA::Buffer *GetGPUWriteBuffer() override { Utils::Logger::LogError("Unable to allocate table GPU buffer"); }
	CUDA::Buffer *GetGPUReadBuffer() const override { Utils::Logger::LogError("Unable to allocate table GPU buffer"); }

	// Printers

	std::string Description() const override;
	std::string DebugDump() const override;

private:
	HorseIR::TableType *m_type = new HorseIR::TableType();

	std::unordered_map<std::string, VectorBuffer *> m_columns;
	unsigned long m_rows = 0;
};

}

