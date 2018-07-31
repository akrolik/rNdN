#include "Runtime/Table.h"

namespace Runtime {

void Table::AddColumn(const std::string& name, List *column)
{
	m_columns.insert({name, column});
}

List *Table::GetColumn(const std::string& name) const
{
	return m_columns.at(name);
}

}
