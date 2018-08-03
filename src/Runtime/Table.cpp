#include "Runtime/Table.h"

#include "Utils/Logger.h"

namespace Runtime {

void Table::AddColumn(const std::string& name, Vector *column)
{
	m_columns.insert({name, column});
}

Vector *Table::GetColumn(const std::string& name) const
{
	if (m_columns.find(name) == m_columns.end())
	{
		Utils::Logger::LogError("Column '" + name + "' not found");
	}
	return m_columns.at(name);
}

void Table::Dump() const
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
