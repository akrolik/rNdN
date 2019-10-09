#include "Runtime/DataBuffers/TableBuffer.h"

#include "HorseIR/Utils/PrettyPrinter.h"

#include "Utils/Logger.h"

namespace Runtime {

TableBuffer::TableBuffer(const std::unordered_map<std::string, VectorBuffer *>& columns) : DataBuffer(DataBuffer::Kind::Table), m_columns(columns)
{
	bool first = true;
	for (const auto& column : columns)
	{
		auto columnRows = column.second->GetElementCount();
		if (first)
		{
			m_rows = columnRows;
			first = false;
		}
		else if (columnRows != m_rows)
		{
			Utils::Logger::LogError("Column length does not match table size [" + std::to_string(columnRows) + " != " + std::to_string(m_rows) + "]");
		}
	}
	m_shape = new Analysis::TableShape(new Analysis::Shape::ConstantSize(m_columns.size()), new Analysis::Shape::ConstantSize(m_rows));
}

VectorBuffer *TableBuffer::GetColumn(const std::string& name) const
{
	if (m_columns.find(name) == m_columns.end())
	{
		Utils::Logger::LogError("Column '" + name + "' not found");
	}
	return m_columns.at(name);
}

std::string TableBuffer::Description() const
{
	std::string description = HorseIR::PrettyPrinter::PrettyString(m_type) + "{";
	bool first = true;
	for (const auto& column : m_columns)
	{
		if (!first)
		{
			description += ", ";
		}
		first = false;
		description += column.first + "=" + column.second->Description();
	}
	return description + "}";
}

std::string TableBuffer::DebugDump() const
{
	std::string string = " ";
	for (const auto& column : m_columns)
	{
		string += column.first + "\t";
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
