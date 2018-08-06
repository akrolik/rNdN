#include "Runtime/DataObjects/DataTable.h"

#include "Utils/Logger.h"

namespace Runtime {

void DataTable::AddColumn(const std::string& name, DataVector *column)
{
	if (column->GetElementCount() != m_size)
	{
		Utils::Logger::LogError("Column length does not match table size [" + std::to_string(column->GetElementCount()) + " != " + std::to_string(m_size) + "]");
	}
	m_columns.insert({name, column});
}

DataVector *DataTable::GetColumn(const std::string& name) const
{
	if (m_columns.find(name) == m_columns.end())
	{
		Utils::Logger::LogError("Column '" + name + "' not found");
	}
	return m_columns.at(name);
}

void DataTable::Dump() const
{
	std::string columnNames = " ";
	for (const auto& column : m_columns)
	{
		columnNames += column.first + "\t";
	}
	Utils::Logger::LogInfo(columnNames, "RESULT");
	Utils::Logger::LogInfo("", "RESULT");

	for (unsigned long i = 0; i < m_size; ++i)
	{
		std::string row;
		for (const auto& column : m_columns)
		{
			row += column.second->Dump(i) + "\t";
		}
		Utils::Logger::LogInfo(row, "RESULT");
	}
}

}
