#pragma once

#include "Runtime/DataObjects/DataObject.h"

#include <string>
#include <unordered_map>

#include "HorseIR/Tree/Types/TableType.h"

#include "Runtime/DataObjects/DataVector.h"

namespace Runtime {

class DataTable : public DataObject
{
public:
	DataTable(unsigned long size) : m_size(size) {}

	const HorseIR::TableType *GetType() const { return m_type; }

	void AddColumn(const std::string& name, DataVector *column);
	DataVector *GetColumn(const std::string& column) const;

	unsigned long GetSize() const { return m_size; }

	void Dump() const override;

private:
	const HorseIR::TableType *m_type = new HorseIR::TableType();

	std::unordered_map<std::string, DataVector *> m_columns;
	unsigned long m_size = 0;
};

}
