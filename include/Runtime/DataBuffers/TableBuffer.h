#pragma once

#include "Runtime/DataBuffers/DataBuffer.h"

#include <string>
#include <utility>
#include <vector>

#include "HorseIR/Analysis/Shape/Shape.h"
#include "HorseIR/Tree/Tree.h"

#include "Runtime/DataBuffers/BufferUtils.h"
#include "Runtime/DataBuffers/ColumnBuffer.h"
#include "Runtime/DataBuffers/VectorBuffer.h"

#include "Libraries/robin_hood.h"

#include "Utils/Logger.h"

namespace Runtime {

class TableBuffer : public DataBuffer
{
public:
	constexpr static DataBuffer::Kind BufferKind = DataBuffer::Kind::Table;

	TableBuffer(const std::vector<std::pair<std::string, ColumnBuffer *>>& columns);
	~TableBuffer() override;

	TableBuffer *Clone() const override;

	// Tag

	void SetTag(const std::string& tag) override;

	// Type/shape

	const HorseIR::TableType *GetType() const override { return m_type; }
	const HorseIR::Analysis::TableShape *GetShape() const override { return m_shape; }

	// Columns
	
	void SetPrimaryKey(VectorBuffer *primaryKey, const robin_hood::unordered_map<std::int32_t, std::int64_t>& primaryMap)
	{
		m_primaryKey = primaryKey; 
		m_primaryMap = primaryMap;
	}
	
	const VectorBuffer *GetPrimaryKey() const { return m_primaryKey; }
	VectorBuffer *GetPrimaryKey() { return m_primaryKey; }

	const robin_hood::unordered_map<std::int32_t, std::int64_t>& GetPrimaryMap() const { return m_primaryMap; }

	const ColumnBuffer *GetColumn(const std::string& column) const;
	ColumnBuffer *GetColumn(const std::string& column);

	const std::vector<std::pair<std::string, ColumnBuffer *>>& GetColumns() const { return m_columns; }

	// CPU/GPU management

	void ValidateCPU() const override;
	void ValidateGPU() const override;

	// Printers

	std::string Description() const override;
	std::string DebugDump(unsigned int indent = 0, bool preindent = false) const override;

	// Clear

	void Clear(ClearMode mode = ClearMode::Zero) override;

private:
	const HorseIR::TableType *m_type = new HorseIR::TableType();
	const HorseIR::Analysis::TableShape *m_shape = nullptr;

	std::vector<std::pair<std::string, ColumnBuffer *>> m_columns;
	robin_hood::unordered_map<std::string, ColumnBuffer *> m_columnMap;

	VectorBuffer *m_primaryKey = nullptr;
	robin_hood::unordered_map<std::int32_t, std::int64_t> m_primaryMap;

	unsigned int m_rows = 0;
};

}

