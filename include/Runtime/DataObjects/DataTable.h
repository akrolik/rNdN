#pragma once

#include "Runtime/DataObjects/DataObject.h"

#include <string>
#include <unordered_map>

#include "HorseIR/Tree/Tree.h"
#include "HorseIR/Utils/PrettyPrinter.h"

#include "Runtime/DataObjects/DataVector.h"

namespace Runtime {

class DataTable : public DataObject
{
public:
	DataTable(unsigned long size) : m_size(size) {}

	HorseIR::TableType *GetType() const { return m_type; }

	void AddColumn(const std::string& name, DataVector *column);
	DataVector *GetColumn(const std::string& column) const;

	unsigned long GetSize() const { return m_size; }

	std::string Description() const override
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

	std::string DebugDump() const override;

private:
	HorseIR::TableType *m_type = new HorseIR::TableType();

	std::unordered_map<std::string, DataVector *> m_columns;
	unsigned long m_size = 0;
};

}
