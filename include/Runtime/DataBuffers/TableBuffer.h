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
	TableBuffer(unsigned long rows) : m_rows(rows)
	{
		m_shape = new Analysis::TableShape(new Analysis::Shape::ConstantSize(0), new Analysis::Shape::ConstantSize(rows));
	}

	HorseIR::TableType *GetType() const override { return m_type; }
	const Analysis::TableShape *GetShape() const override { return m_shape; }

	// Columns

	//TODO: Update table size
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
	Analysis::TableShape *m_shape = nullptr;

	std::unordered_map<std::string, VectorBuffer *> m_columns;
	unsigned long m_rows = 0;
};

}

