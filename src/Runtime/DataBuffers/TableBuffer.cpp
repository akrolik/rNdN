#include "Runtime/DataBuffers/TableBuffer.h"

#include <iomanip>
#include <iostream>

#include "HorseIR/Utils/PrettyPrinter.h"

#include "Utils/Logger.h"

namespace Runtime {

TableBuffer::TableBuffer(const std::vector<std::pair<std::string, ColumnBuffer *>>& columns) : DataBuffer(DataBuffer::Kind::Table), m_columns(columns)
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

ColumnBuffer *TableBuffer::GetColumn(const std::string& name) const
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
	std::stringstream string;
	for (const auto& [name, data] : m_columns)
	{
		string << std::left;
		if (HorseIR::TypeUtils::IsDatetimeType(data->GetType()))
		{
			string << std::setw(30);
		}
		else
		{
			string << std::setw(20);
		}
		string << name;
	}
	string << std::endl;
	for (auto i = 0ul; i < m_rows; ++i)
	{
		for (const auto& [name, data] : m_columns)
		{
			string << std::left;
			if (HorseIR::TypeUtils::IsDatetimeType(data->GetType()))
			{
				string << std::setw(30);
			}
			else
			{
				string << std::setw(20);
			}
		       	string << data->DebugDump(i);
		}
		string << std::endl;
	}
	return string.str();
}

}
