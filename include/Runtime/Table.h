#pragma once

#include "Runtime/DataObject.h"

#include <string>
#include <unordered_map>

#include "Runtime/Vector.h"

namespace Runtime {

class Table : public DataObject
{
public:
	Table(const std::string& name, unsigned long size) : m_name(name), m_size(size) {}

	void AddColumn(const std::string& name, Vector *column);
	Vector *GetColumn(const std::string& column) const;

	const std::string& GetName() { return m_name; }
	unsigned long GetSize() const { return m_size; }

	void Dump() const override;

private:
	std::string m_name;

	std::unordered_map<std::string, Vector *> m_columns;
	unsigned long m_size = 0;
};

}
