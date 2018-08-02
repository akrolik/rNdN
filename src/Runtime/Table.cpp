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

}
