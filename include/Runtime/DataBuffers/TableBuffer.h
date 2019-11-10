#pragma once

#include "Runtime/DataBuffers/DataBuffer.h"

#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include "Analysis/Shape/Shape.h"

#include "HorseIR/Tree/Tree.h"

#include "Runtime/DataBuffers/ColumnBuffer.h"
#include "Runtime/DataBuffers/VectorBuffer.h"

namespace Runtime {

class TableBuffer : public DataBuffer
{
public:
	constexpr static DataBuffer::Kind BufferKind = DataBuffer::Kind::Table;

	TableBuffer(const std::vector<std::pair<std::string, ColumnBuffer *>>& columns);
	~TableBuffer() override;

	const HorseIR::TableType *GetType() const override { return m_type; }
	const Analysis::TableShape *GetShape() const override { return m_shape; }

	// Columns
	
	void SetPrimaryKey(VectorBuffer *primaryKey, const std::unordered_map<std::int64_t, std::int64_t>& primaryMap)
	{
		m_primaryKey = primaryKey; 
		m_primaryMap = primaryMap;
	}
	
	VectorBuffer *GetPrimaryKey() const { return m_primaryKey; }
	const std::unordered_map<std::int64_t, std::int64_t>& GetPrimaryMap() const { return m_primaryMap; }

	ColumnBuffer *GetColumn(const std::string& column) const;

	// CPU/GPU management

	CUDA::Buffer *GetGPUWriteBuffer() override { Utils::Logger::LogError("Unable to allocate table GPU buffer"); }
	CUDA::Buffer *GetGPUReadBuffer() const override { Utils::Logger::LogError("Unable to allocate table GPU buffer"); }

	size_t GetGPUBufferSize() const override { return 0; }

	// Printers

	std::string Description() const override;
	std::string DebugDump() const override;

private:
	const HorseIR::TableType *m_type = new HorseIR::TableType();
	const Analysis::TableShape *m_shape = nullptr;

	std::vector<std::pair<std::string, ColumnBuffer *>> m_columns;
	std::unordered_map<std::string, ColumnBuffer *> m_columnMap;

	VectorBuffer *m_primaryKey = nullptr;
	std::unordered_map<std::int64_t, std::int64_t> m_primaryMap;

	unsigned int m_rows = 0;
};

}

