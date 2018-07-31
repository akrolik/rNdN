#pragma once

#include "Runtime/DataObject.h"

#include <string>
#include <unordered_map>

#include "Runtime/List.h"

namespace Runtime {

class Table : public DataObject
{
public:
	Table(const std::string& name, unsigned long size) : m_name(name), m_size(size) {}

	void AddColumn(const std::string& name, List *column);
	List *GetColumn(const std::string& column) const;

	const std::string& GetName() { return m_name; }
	unsigned long GetSize() const { return m_size; }

private:
	std::string m_name;

	std::unordered_map<std::string, List *> m_columns;
	unsigned long m_size = 0;
};

}
