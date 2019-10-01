#pragma once

#include <string>
#include <unordered_map>

#include "HorseIR/Tree/Tree.h"

#include "Runtime/DataObjects/DataTable.h"

namespace Runtime {

class DataRegistry
{
public:
	template<typename T>
	static void LoadDebugData(DataTable *table, const HorseIR::BasicType *type, unsigned long size);

	void LoadDebugData();

	void AddTable(const std::string& name, DataTable *table);
	DataTable *GetTable(const std::string& name) const;

	void LoadFile(const std::string& filename);

private:
	std::unordered_map<std::string, DataTable *> m_registry;
};

}
