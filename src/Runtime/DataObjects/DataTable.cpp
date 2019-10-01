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

std::string DataTable::DebugDump() const
{
	std::string string = " ";
	for (const auto& column : m_columns)
	{
		string += column.first + "\t";
	}
	string += "\n";
	for (auto i = 0ul; i < m_size; ++i)
	{
		for (const auto& column : m_columns)
		{
			string += column.second->DebugDump(i) + "\t";
		}
		string += "\n";
	}
	return string;
}

}
