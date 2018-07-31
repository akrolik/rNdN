#pragma once

#include <string>
#include <unordered_map>

#include "HorseIR/Tree/Types/Type.h"

#include "Runtime/Table.h"

namespace Runtime {

class DataRegistry
{
public:
	template<typename T>
	static void LoadDebugData(Table *table, const HorseIR::Type *type, unsigned long size);

	void LoadDebugData();

	void AddTable(const std::string& name, Table *table);
	Table *GetTable(const std::string& name) const;

	void LoadFile(const std::string& filename);

private:
	std::unordered_map<std::string, Table *> m_registry;
};

}
