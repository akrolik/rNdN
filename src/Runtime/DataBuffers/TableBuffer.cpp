#include "Runtime/DataBuffers/TableBuffer.h"

#include "HorseIR/Utils/PrettyPrinter.h"

#include "Utils/Logger.h"

namespace Runtime {

TableBuffer::TableBuffer(const std::vector<std::pair<std::string, VectorBuffer *>>& columns) : DataBuffer(DataBuffer::Kind::Table), m_columns(columns)
{
	bool first = true;
	for (const auto& [name, data] : columns)
	{
		auto columnRows = data->GetElementCount();
		if (first)
		{
			m_rows = columnRows;
			first = false;
		}
		else if (columnRows != m_rows)
		{
			Utils::Logger::LogError("Column '" + name + "' length does not match table size [" + std::to_string(columnRows) + " != " + std::to_string(m_rows) + "]");
		}

		// Add the column to the map

		if (m_columnMap.find(name) != m_columnMap.end())
		{
			Utils::Logger::LogError("Duplicate column '" + name + "'");
		}
		m_columnMap[name] = data;
	}
	m_shape = new Analysis::TableShape(new Analysis::Shape::ConstantSize(m_columns.size()), new Analysis::Shape::ConstantSize(m_rows));
}

TableBuffer::~TableBuffer()
{
	delete m_type;
	delete m_shape;
}

VectorBuffer *TableBuffer::GetColumn(const std::string& name) const
{
	if (m_columnMap.find(name) == m_columnMap.end())
	{
		Utils::Logger::LogError("Column '" + name + "' not found");
	}
	return m_columnMap.at(name);
}

std::string TableBuffer::Description() const
{
	std::string description = HorseIR::PrettyPrinter::PrettyString(m_type) + "{";
	bool first = true;
	for (const auto& [name, data] : m_columns)
	{
		if (!first)
		{
			description += ", ";
		}
		first = false;
		description += name + "=" + data->Description();
	}
	return description + "}";
}

std::string TableBuffer::DebugDump() const
{
	std::string string = " ";
	for (const auto& [name, data] : m_columns)
	{
		string += name + "\t";
	}
	string += "\n";
	for (auto i = 0ul; i < m_rows; ++i)
	{
		for (const auto& column : m_columns)
		{
			string += column.second->GetCPUReadBuffer()->DebugDump(i) + "\t";
		}
		string += "\n";
	}
	return string;
}

}
