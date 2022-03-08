#pragma once

#include <string>
#include <utility>
#include <vector>

#include "HorseIR/Tree/Tree.h"

#include "Libraries/robin_hood.h"

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
	void LoadTPCHData(unsigned int scale = 1);

	void AddTable(const std::string& db, const std::string& name, TableBuffer *table);
	TableBuffer *GetTable(const std::string& name) const;

	void LoadFile(const std::string& filename);

private:
	std::int32_t EpochTime(char *date) const;
	std::string GetTPCHPath(const std::string& table) const;

	void LoadTPCHNationTable();
	void LoadTPCHRegionTable();
	void LoadTPCHPartTable(unsigned int scale);
	void LoadTPCHSupplierTable(unsigned int scale);
	void LoadTPCHPartSupplierTable(unsigned int scale);
	void LoadTPCHCustomerTable(unsigned int scale);
	void LoadTPCHOrderTable(unsigned int scale);
	void LoadTPCHLineItemTable(unsigned int scale);

	robin_hood::unordered_map<std::string, TableBuffer *> m_registry;
};

}
