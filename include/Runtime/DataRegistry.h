#pragma once

#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include "HorseIR/Tree/Tree.h"

namespace Runtime {

class ColumnBuffer;
class TableBuffer;

class DataRegistry
{
public:
	template<typename T>
	static void LoadDebugData(std::vector<std::pair<std::string, ColumnBuffer *>>& columns, const HorseIR::BasicType *type, unsigned long size);
	static void LoadDateDebugData(std::vector<std::pair<std::string, ColumnBuffer *>>& columns, unsigned long size);
	static void LoadStringDebugData(std::vector<std::pair<std::string, ColumnBuffer *>>& columns, unsigned long size);

	void LoadDebugData();
	void LoadTPCHData();

	void AddTable(const std::string& name, TableBuffer *table);
	TableBuffer *GetTable(const std::string& name) const;

	void LoadFile(const std::string& filename);

private:
	std::int32_t EpochTime(char *date) const;

	void LoadTPCHNationTable();
	void LoadTPCHRegionTable();
	void LoadTPCHPartTable();
	void LoadTPCHSupplierTable();
	void LoadTPCHPartSupplierTable();
	void LoadTPCHCustomerTable();
	void LoadTPCHOrderTable();
	void LoadTPCHLineItemTable();

	std::unordered_map<std::string, TableBuffer *> m_registry;
};

}
