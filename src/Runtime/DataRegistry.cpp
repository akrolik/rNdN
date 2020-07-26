#include "Runtime/DataRegistry.h"

#include <stdlib.h>

#include "CUDA/Vector.h"

#include "HorseIR/Utils/PrettyPrinter.h"

#include "Libraries/csv.h"

#include "Runtime/DataBuffers/ColumnBuffer.h"
#include "Runtime/DataBuffers/EnumerationBuffer.h"
#include "Runtime/DataBuffers/TableBuffer.h"
#include "Runtime/DataBuffers/VectorBuffer.h"
#include "Runtime/StringBucket.h"

#include "Utils/Chrono.h"
#include "Utils/Date.h"
#include "Utils/Logger.h"
#include "Utils/Progress.h"

namespace Runtime {

template<typename T>
void DataRegistry::LoadDebugData(std::vector<std::pair<std::string, ColumnBuffer *>>& columns, const HorseIR::BasicType *type, unsigned long size)
{
	CUDA::Vector<T> zeros(size);
	CUDA::Vector<T> ones(size);
	CUDA::Vector<T> asc(size);
	CUDA::Vector<T> desc(size);

	CUDA::Vector<T> fkey(size);
	CUDA::Vector<T> fval(size);
	CUDA::Vector<std::int64_t> findexes(size);
	
	for (auto i = 0u; i < size; ++i)
	{
		zeros[i] = 0;
		ones[i] = 1;
		asc[i] = i;
		desc[i] = size - i - 1;

		fkey[i] = i;
		fval[i] = (i / 2) * 2;
		findexes[i] = (i / 2) * 2;
	}

	auto name = HorseIR::PrettyPrinter::PrettyString(type);
	columns.push_back({"zeros_" + name, new TypedVectorBuffer(new TypedVectorData(type, std::move(zeros)))});
	columns.push_back({"ones_" + name, new TypedVectorBuffer(new TypedVectorData(type, std::move(ones)))});
	columns.push_back({"asc_" + name, new TypedVectorBuffer(new TypedVectorData(type, std::move(asc)))});
	columns.push_back({"desc_" + name, new TypedVectorBuffer(new TypedVectorData(type, std::move(desc)))});

	auto foreignBuffer = new TypedVectorBuffer(new TypedVectorData(type, std::move(fkey)));
	columns.push_back({"fkey_" + name, foreignBuffer});
	columns.push_back({"enum_" + name, new EnumerationBuffer(foreignBuffer,
		new TypedVectorBuffer(new TypedVectorData(type, std::move(fval))),
		new TypedVectorBuffer(new TypedVectorData(new HorseIR::BasicType(HorseIR::BasicType::BasicKind::Int64), std::move(findexes)))
	)});
}

void DataRegistry::LoadDateDebugData(std::vector<std::pair<std::string, ColumnBuffer *>>& columns, unsigned long size)
{
	CUDA::Vector<std::int32_t> dates(size);
	CUDA::Vector<std::int64_t> times(size);
	CUDA::Vector<std::int64_t> datetimes(size);

	for (auto i = 0u; i < size; ++i)
	{
		auto year = 2000 + (i / 36);
		auto month = (i % 36) / 3 + 1;
		auto day = (i % 3);

		auto hour = (i % 24);
		auto minute = (i % 60);
		auto second = (i % 60);
		auto millisecond = (i % 1000);

		dates[i] = Utils::Date::EpochTime_day(year, month, day);
		times[i] = Utils::Date::ExtendedEpochTime_time(hour, minute, second, millisecond);
		datetimes[i] = Utils::Date::ExtendedEpochTime(year, month, day, hour, minute, second, millisecond);
	}

	columns.push_back({"dates", new TypedVectorBuffer(new TypedVectorData(new HorseIR::BasicType(HorseIR::BasicType::BasicKind::Date), std::move(dates)))});
	columns.push_back({"times", new TypedVectorBuffer(new TypedVectorData(new HorseIR::BasicType(HorseIR::BasicType::BasicKind::Time), std::move(times)))});
	columns.push_back({"datetimes", new TypedVectorBuffer(new TypedVectorData(new HorseIR::BasicType(HorseIR::BasicType::BasicKind::Datetime), std::move(datetimes)))});
}

void DataRegistry::LoadStringDebugData(std::vector<std::pair<std::string, ColumnBuffer *>>& columns, unsigned long size)
{
	CUDA::Vector<std::uint64_t> strings(size);
	CUDA::Vector<std::uint64_t> symbols(size);

	for (auto i = 0u; i < size; ++i)
	{
		auto numberString = std::to_string(i);
		auto paddedString = std::string(5 - numberString.length(), '0') + numberString;
		strings[i] = StringBucket::HashString("String#" + paddedString);
		symbols[i] = StringBucket::HashString("Symbol#" + paddedString);
	}

	columns.push_back({"strings", new TypedVectorBuffer(new TypedVectorData(new HorseIR::BasicType(HorseIR::BasicType::BasicKind::String), std::move(strings)))});
	columns.push_back({"symbols", new TypedVectorBuffer(new TypedVectorData(new HorseIR::BasicType(HorseIR::BasicType::BasicKind::Symbol), std::move(symbols)))});
}

void DataRegistry::LoadDebugData()
{
	Utils::Logger::LogSection("Loading debug data");

	auto timeData_start = Utils::Chrono::Start("Load debug data");

	for (auto size = 1ul; size <= 64 * 1024; size <<= 1)
	{
		std::vector<std::pair<std::string, ColumnBuffer *>> columns;

		std::unordered_map<std::int32_t, std::int64_t> primaryMap;
		CUDA::Vector<std::int64_t> primaryKey(size);
		for (auto i = 0u; i < size; ++i)
		{
			primaryMap[i] = i;
			primaryKey[i] = i;
		}
		auto primaryBuffer = new TypedVectorBuffer(new TypedVectorData(new HorseIR::BasicType(HorseIR::BasicType::BasicKind::Int32), std::move(primaryKey)));
		columns.push_back({"pkey", primaryBuffer});

		LoadDebugData<std::int8_t>(columns, new HorseIR::BasicType(HorseIR::BasicType::BasicKind::Boolean), size);
		LoadDebugData<std::int8_t>(columns, new HorseIR::BasicType(HorseIR::BasicType::BasicKind::Int8), size);
		LoadDebugData<std::int16_t>(columns, new HorseIR::BasicType(HorseIR::BasicType::BasicKind::Int16), size);
		LoadDebugData<std::int32_t>(columns, new HorseIR::BasicType(HorseIR::BasicType::BasicKind::Int32), size);
		LoadDebugData<std::int64_t>(columns, new HorseIR::BasicType(HorseIR::BasicType::BasicKind::Int64), size);
		LoadDebugData<float>(columns, new HorseIR::BasicType(HorseIR::BasicType::BasicKind::Float32), size);
		LoadDebugData<double>(columns, new HorseIR::BasicType(HorseIR::BasicType::BasicKind::Float64), size);
		LoadDateDebugData(columns, size);
		LoadStringDebugData(columns, size);

		auto table = new TableBuffer(columns);
		table->SetPrimaryKey(primaryBuffer, primaryMap);

		AddTable("debug", "debug_" + std::to_string(size), table);

		if (Utils::Options::Present(Utils::Options::Opt_Print_load))
		{
			Utils::Logger::LogInfo("Loaded table 'debug_" + std::to_string(size) + "'");
		}
	}

	Utils::Chrono::End(timeData_start);
}

void DataRegistry::LoadTPCHData()
{
	Utils::Logger::LogSection("Loading TPC-H data");

	auto timeData_start = Utils::Chrono::Start("Load TPC-H data");

	LoadTPCHRegionTable();
	LoadTPCHNationTable();
	LoadTPCHSupplierTable();
	LoadTPCHPartTable();
	LoadTPCHPartSupplierTable();
	LoadTPCHCustomerTable();
	LoadTPCHOrderTable();
	LoadTPCHLineItemTable();

	Utils::Chrono::End(timeData_start);
}

std::string DataRegistry::GetTPCHPath(const std::string& table) const
{
	return Utils::Options::Get<std::string>(Utils::Options::Opt_Data_path_tpch) + "/" + table;
}

std::int32_t DataRegistry::EpochTime(char *date) const
{
	int year, month, day;
	sscanf(date, "%d-%d-%d", &year, &month, &day);
	return Utils::Date::EpochTime_day(year, month, day);
}

void DataRegistry::LoadTPCHNationTable()
{
	// CREATE TABLE NATION (
	//     N_NATIONKEY  INTEGER NOT NULL,
	//     N_NAME       CHAR(25) NOT NULL,
	//     N_REGIONKEY  INTEGER NOT NULL,
	//     N_COMMENT    VARCHAR(152)
	// );
	auto size = 25u;

	std::unordered_map<std::int32_t, std::int64_t> primaryMap;

	CUDA::Vector<std::int32_t> nationKey(size); // PKey
	CUDA::Vector<std::uint64_t> name(size);
	CUDA::Vector<std::int32_t> regionVal(size); // FKey[region] value
	CUDA::Vector<std::int64_t> regionKey(size); // FKey[region]
	CUDA::Vector<std::uint64_t> comment(size);

	io::CSVReader<5, io::trim_chars<' ', '\t'>, io::no_quote_escape<'|'>> lineReader(GetTPCHPath("nation.tbl"));
	char *n_nationKey, *n_name, *n_regionKey, *n_comment, *n_end;

	const auto& regionTable = GetTable("region");
	const auto& regionForeignKey = regionTable->GetPrimaryKey();
	const auto& regionForeignMap = regionTable->GetPrimaryMap();

	auto progress = Utils::Progress::Start("Loading table 'nation'", Utils::Options::Present(Utils::Options::Opt_Print_load));
	auto count = 0u;

	while (lineReader.read_row(n_nationKey, n_name, n_regionKey, n_comment, n_end))
	{
		primaryMap[atoi(n_nationKey)] = count;

		nationKey[count] = atoi(n_nationKey);
		name[count] = StringBucket::HashString(n_name);
		regionVal[count] = atoi(n_regionKey);
		regionKey[count] = regionForeignMap.at(atoi(n_regionKey));
		comment[count] = StringBucket::HashString(n_comment);

		count++;
		progress.Update(count, size);
	}
	progress.Complete();

	std::vector<std::pair<std::string, ColumnBuffer *>> columns;

	auto primaryKey = new TypedVectorBuffer(new TypedVectorData(new HorseIR::BasicType(HorseIR::BasicType::BasicKind::Int32), std::move(nationKey)));
	columns.push_back({"n_nationkey", primaryKey});
	columns.push_back({"n_name", new TypedVectorBuffer(new TypedVectorData(new HorseIR::BasicType(HorseIR::BasicType::BasicKind::Symbol), std::move(name)))});
	columns.push_back({"n_regionkey", new EnumerationBuffer(regionForeignKey,
		new TypedVectorBuffer(new TypedVectorData(new HorseIR::BasicType(HorseIR::BasicType::BasicKind::Int32), std::move(regionVal))),
		new TypedVectorBuffer(new TypedVectorData(new HorseIR::BasicType(HorseIR::BasicType::BasicKind::Int64), std::move(regionKey)))
	)});
	columns.push_back({"n_comment", new TypedVectorBuffer(new TypedVectorData(new HorseIR::BasicType(HorseIR::BasicType::BasicKind::String), std::move(comment)))});

	auto table = new TableBuffer(columns);
	table->SetPrimaryKey(primaryKey, primaryMap);
	AddTable("tpch", "nation", table);
}                       

void DataRegistry::LoadTPCHRegionTable()
{
	// CREATE TABLE REGION (
	//     R_REGIONKEY  INTEGER NOT NULL,
	//     R_NAME       CHAR(25) NOT NULL,
	//     R_COMMENT    VARCHAR(152)
	// );
	auto size = 5u;

	std::unordered_map<std::int32_t, std::int64_t> primaryMap;

	CUDA::Vector<std::int32_t> regionKey(size); // PKey
	CUDA::Vector<std::uint64_t> name(size);
	CUDA::Vector<std::uint64_t> comment(size);

	io::CSVReader<4, io::trim_chars<' ', '\t'>, io::no_quote_escape<'|'>> lineReader(GetTPCHPath("region.tbl"));
	char *r_regionKey, *r_name, *r_comment, *r_end;

	auto progress = Utils::Progress::Start("Loading table 'region'", Utils::Options::Present(Utils::Options::Opt_Print_load));
	auto count = 0u;

	while (lineReader.read_row(r_regionKey, r_name, r_comment, r_end))
	{
		primaryMap[atoi(r_regionKey)] = count;

		regionKey[count] = atoi(r_regionKey);
		name[count] = StringBucket::HashString(r_name);
		comment[count] = StringBucket::HashString(r_comment);

		count++;
		progress.Update(count, size);
	}
	progress.Complete();

	std::vector<std::pair<std::string, ColumnBuffer *>> columns;

	auto primaryKey = new TypedVectorBuffer(new TypedVectorData(new HorseIR::BasicType(HorseIR::BasicType::BasicKind::Int32), std::move(regionKey)));
	columns.push_back({"r_regionKey", primaryKey});
	columns.push_back({"r_name", new TypedVectorBuffer(new TypedVectorData(new HorseIR::BasicType(HorseIR::BasicType::BasicKind::Symbol), std::move(name)))});
	columns.push_back({"r_comment", new TypedVectorBuffer(new TypedVectorData(new HorseIR::BasicType(HorseIR::BasicType::BasicKind::String), std::move(comment)))});

	auto table = new TableBuffer(columns);
	table->SetPrimaryKey(primaryKey, primaryMap);
	AddTable("tpch", "region", table);
}

void DataRegistry::LoadTPCHPartTable()
{
	// CREATE TABLE PART (
	//     P_PARTKEY     INTEGER NOT NULL,
	//     P_NAME        VARCHAR(55) NOT NULL,
	//     P_MFGR        CHAR(25) NOT NULL,
	//     P_BRAND       CHAR(10) NOT NULL,
	//     P_TYPE        VARCHAR(25) NOT NULL,
	//     P_SIZE        INTEGER NOT NULL,
	//     P_CONTAINER   CHAR(10) NOT NULL,
	//     P_RETAILPRICE DECIMAL(15,2) NOT NULL,
	//     P_COMMENT     VARCHAR(23) NOT NULL
	// );
	auto _size = 200000u;

	std::unordered_map<std::int32_t, std::int64_t> primaryMap;

	CUDA::Vector<std::int32_t> partKey(_size); // PKey
	CUDA::Vector<std::uint64_t> name(_size);
	CUDA::Vector<std::uint64_t> mfgr(_size);
	CUDA::Vector<std::uint64_t> brand(_size);
	CUDA::Vector<std::uint64_t> type(_size);
	CUDA::Vector<std::int32_t> size(_size);
	CUDA::Vector<std::uint64_t> container(_size);
	CUDA::Vector<double> retailPrice(_size);
	CUDA::Vector<std::uint64_t> comment(_size);

	io::CSVReader<10, io::trim_chars<' ', '\t'>, io::no_quote_escape<'|'>> lineReader(GetTPCHPath("part.tbl"));
	char *p_partKey, *p_name, *p_mfgr, *p_brand, *p_type, *p_size, *p_container, *p_retailPrice, *p_comment, *p_end;

	auto progress = Utils::Progress::Start("Loading table 'part'", Utils::Options::Present(Utils::Options::Opt_Print_load));
	auto count = 0u;

	while (lineReader.read_row(
		p_partKey, p_name, p_mfgr, p_brand, p_type, p_size, p_container, p_retailPrice, p_comment, p_end
	))
	{
		primaryMap[atoi(p_partKey)] = count;

		partKey[count] = atoi(p_partKey);
		name[count] = StringBucket::HashString(p_name);
		mfgr[count] = StringBucket::HashString(p_mfgr);
		brand[count] = StringBucket::HashString(p_brand);
		type[count] = StringBucket::HashString(p_type);
		size[count] = atoi(p_size);
		container[count] = StringBucket::HashString(p_container);
		retailPrice[count] = atof(p_retailPrice);
		comment[count] = StringBucket::HashString(p_comment);

		count++;
		progress.Update(count, _size);
	}
	progress.Complete();

	std::vector<std::pair<std::string, ColumnBuffer *>> columns;

	auto primaryKey = new TypedVectorBuffer(new TypedVectorData(new HorseIR::BasicType(HorseIR::BasicType::BasicKind::Int32), std::move(partKey)));
	columns.push_back({"p_partkey", primaryKey});
	columns.push_back({"p_name", new TypedVectorBuffer(new TypedVectorData(new HorseIR::BasicType(HorseIR::BasicType::BasicKind::Symbol), std::move(name)))});
	columns.push_back({"p_mfgr", new TypedVectorBuffer(new TypedVectorData(new HorseIR::BasicType(HorseIR::BasicType::BasicKind::Symbol), std::move(mfgr)))});
	columns.push_back({"p_brand", new TypedVectorBuffer(new TypedVectorData(new HorseIR::BasicType(HorseIR::BasicType::BasicKind::Symbol), std::move(brand)))});
	columns.push_back({"p_type", new TypedVectorBuffer(new TypedVectorData(new HorseIR::BasicType(HorseIR::BasicType::BasicKind::Symbol), std::move(type)))});
	columns.push_back({"p_size", new TypedVectorBuffer(new TypedVectorData(new HorseIR::BasicType(HorseIR::BasicType::BasicKind::Int32), std::move(size)))});
	columns.push_back({"p_container", new TypedVectorBuffer(new TypedVectorData(new HorseIR::BasicType(HorseIR::BasicType::BasicKind::Symbol), std::move(container)))});
	columns.push_back({"p_retailprice", new TypedVectorBuffer(new TypedVectorData(new HorseIR::BasicType(HorseIR::BasicType::BasicKind::Float64), std::move(retailPrice)))});
	columns.push_back({"p_comment", new TypedVectorBuffer(new TypedVectorData(new HorseIR::BasicType(HorseIR::BasicType::BasicKind::String), std::move(comment)))});
	
	auto table = new TableBuffer(columns);
	table->SetPrimaryKey(primaryKey, primaryMap);
	AddTable("tpch", "part", table);
}

void DataRegistry::LoadTPCHSupplierTable()
{
	// CREATE TABLE SUPPLIER (
	//     S_SUPPKEY     INTEGER NOT NULL,
	//     S_NAME        CHAR(25) NOT NULL,
	//     S_ADDRESS     VARCHAR(40) NOT NULL,
	//     S_NATIONKEY   INTEGER NOT NULL,
	//     S_PHONE       CHAR(15) NOT NULL,
	//     S_ACCTBAL     DECIMAL(15,2) NOT NULL,
	//     S_COMMENT     VARCHAR(101) NOT NULL
	// );
	auto size = 10000u;

	std::unordered_map<std::int32_t, std::int64_t> primaryMap;

	CUDA::Vector<std::int32_t> supplierKey(size); // PKey
	CUDA::Vector<std::uint64_t> name(size);
	CUDA::Vector<std::uint64_t> address(size);
	CUDA::Vector<std::int32_t> nationVal(size); // FKey[nation] value
	CUDA::Vector<std::int64_t> nationKey(size); // FKey[nation]
	CUDA::Vector<std::uint64_t> phone(size);
	CUDA::Vector<double> balance(size);
	CUDA::Vector<std::uint64_t> comment(size);

	io::CSVReader<8, io::trim_chars<' ', '\t'>, io::no_quote_escape<'|'>> lineReader(GetTPCHPath("supplier.tbl"));
	char *s_suppKey, *s_name, *s_address, *s_nationKey, *s_phone, *s_acctBal, *s_comment, *s_end;

	const auto& nationTable = GetTable("nation");
	const auto& nationForeignKey = nationTable->GetPrimaryKey();
	const auto& nationForeignMap = nationTable->GetPrimaryMap();

	auto progress = Utils::Progress::Start("Loading table 'supplier'", Utils::Options::Present(Utils::Options::Opt_Print_load));
	auto count = 0u;

	while (lineReader.read_row(
		s_suppKey, s_name, s_address, s_nationKey, s_phone, s_acctBal, s_comment, s_end
	))
	{
		primaryMap[atoi(s_suppKey)] = count;

		supplierKey[count] = atoi(s_suppKey);
		name[count] = StringBucket::HashString(s_name);
		address[count] = StringBucket::HashString(s_address);
		nationVal[count] = atoi(s_nationKey);
		nationKey[count] = nationForeignMap.at(atoi(s_nationKey));
		phone[count] = StringBucket::HashString(s_phone);
		balance[count] = atof(s_acctBal);
		comment[count] = StringBucket::HashString(s_comment);

		count++;
		progress.Update(count, size);
	}
	progress.Complete();

	std::vector<std::pair<std::string, ColumnBuffer *>> columns;

	auto primaryKey = new TypedVectorBuffer(new TypedVectorData(new HorseIR::BasicType(HorseIR::BasicType::BasicKind::Int32), std::move(supplierKey)));
	columns.push_back({"s_suppkey", primaryKey});
	columns.push_back({"s_name", new TypedVectorBuffer(new TypedVectorData(new HorseIR::BasicType(HorseIR::BasicType::BasicKind::Symbol), std::move(name)))});
	columns.push_back({"s_address", new TypedVectorBuffer(new TypedVectorData(new HorseIR::BasicType(HorseIR::BasicType::BasicKind::String), std::move(address)))});
	columns.push_back({"s_nationkey", new EnumerationBuffer(nationForeignKey,
		new TypedVectorBuffer(new TypedVectorData(new HorseIR::BasicType(HorseIR::BasicType::BasicKind::Int32), std::move(nationVal))),
		new TypedVectorBuffer(new TypedVectorData(new HorseIR::BasicType(HorseIR::BasicType::BasicKind::Int64), std::move(nationKey)))
	)});
	columns.push_back({"s_phone", new TypedVectorBuffer(new TypedVectorData(new HorseIR::BasicType(HorseIR::BasicType::BasicKind::Symbol), std::move(phone)))});
	columns.push_back({"s_acctbal", new TypedVectorBuffer(new TypedVectorData(new HorseIR::BasicType(HorseIR::BasicType::BasicKind::Float64), std::move(balance)))});
	columns.push_back({"s_comment", new TypedVectorBuffer(new TypedVectorData(new HorseIR::BasicType(HorseIR::BasicType::BasicKind::String), std::move(comment)))});
	
	auto table = new TableBuffer(columns);
	table->SetPrimaryKey(primaryKey, primaryMap);
	AddTable("tpch", "supplier", table);
}

void DataRegistry::LoadTPCHPartSupplierTable()
{
	// CREATE TABLE PARTSUPP (
	//     PS_PARTKEY     INTEGER NOT NULL,
	//     PS_SUPPKEY     INTEGER NOT NULL,
	//     PS_AVAILQTY    INTEGER NOT NULL,
	//     PS_SUPPLYCOST  DECIMAL(15,2)  NOT NULL,
	//     PS_COMMENT     VARCHAR(199) NOT NULL
	// );
	auto size = 800000u;

	//TODO: Primary key
	CUDA::Vector<std::int32_t> partVal(size); // FKey[part] value, PKey
	CUDA::Vector<std::int64_t> partKey(size); // FKey[part], PKey
	CUDA::Vector<std::int32_t> supplierVal(size); // FKey[supplier] value, PKey
	CUDA::Vector<std::int64_t> supplierKey(size); // FKey[supplier], PKey

	CUDA::Vector<std::int32_t> availableQuantity(size);
	CUDA::Vector<double> supplyCost(size);
	CUDA::Vector<std::uint64_t> comment(size);

	io::CSVReader<6, io::trim_chars<' ', '\t'>, io::no_quote_escape<'|'>> lineReader(GetTPCHPath("partsupp.tbl"));
	char *ps_partKey, *ps_suppKey, *ps_availQty, *ps_supplyCost, *ps_comment, *ps_end;

	const auto& partTable = GetTable("part");
	const auto& supplierTable = GetTable("supplier");

	const auto& partForeignMap = partTable->GetPrimaryMap();
	const auto& partForeignKey = partTable->GetPrimaryKey();

	const auto& supplierForeignMap = supplierTable->GetPrimaryMap();
	const auto& supplierForeignKey = supplierTable->GetPrimaryKey();

	auto progress = Utils::Progress::Start("Loading table 'partsupp'", Utils::Options::Present(Utils::Options::Opt_Print_load));
	auto count = 0u;

	while (lineReader.read_row(
		ps_partKey, ps_suppKey, ps_availQty, ps_supplyCost, ps_comment, ps_end
	))
	{
		partVal[count] = atoi(ps_partKey);
		partKey[count] = partForeignMap.at(atoi(ps_partKey));

		supplierVal[count] = atoi(ps_suppKey);
		supplierKey[count] = partForeignMap.at(atoi(ps_suppKey));

		availableQuantity[count] = atoi(ps_availQty);
		supplyCost[count] = atof(ps_supplyCost);
		comment[count] = StringBucket::HashString(ps_comment);

		count++;
		progress.Update(count, size);
	}
	progress.Complete();

	std::vector<std::pair<std::string, ColumnBuffer *>> columns;

	columns.push_back({"ps_partkey", new EnumerationBuffer(partForeignKey,
		new TypedVectorBuffer(new TypedVectorData(new HorseIR::BasicType(HorseIR::BasicType::BasicKind::Int32), std::move(partVal))),
		new TypedVectorBuffer(new TypedVectorData(new HorseIR::BasicType(HorseIR::BasicType::BasicKind::Int64), std::move(partKey)))
	)});
	columns.push_back({"ps_suppkey", new EnumerationBuffer(supplierForeignKey,
		new TypedVectorBuffer(new TypedVectorData(new HorseIR::BasicType(HorseIR::BasicType::BasicKind::Int32), std::move(supplierVal))),
		new TypedVectorBuffer(new TypedVectorData(new HorseIR::BasicType(HorseIR::BasicType::BasicKind::Int64), std::move(supplierKey)))
	)});

	columns.push_back({"ps_availqty", new TypedVectorBuffer(new TypedVectorData(new HorseIR::BasicType(HorseIR::BasicType::BasicKind::Int32), std::move(availableQuantity)))});
	columns.push_back({"ps_supplycost", new TypedVectorBuffer(new TypedVectorData(new HorseIR::BasicType(HorseIR::BasicType::BasicKind::Float64), std::move(supplyCost)))});
	columns.push_back({"ps_comment", new TypedVectorBuffer(new TypedVectorData(new HorseIR::BasicType(HorseIR::BasicType::BasicKind::String), std::move(comment)))});
	
	AddTable("tpch", "partsupp", new TableBuffer(columns));
}

void DataRegistry::LoadTPCHCustomerTable()
{
	// CREATE TABLE CUSTOMER (
	//     C_CUSTKEY     INTEGER NOT NULL,
	//     C_NAME        VARCHAR(25) NOT NULL,
	//     C_ADDRESS     VARCHAR(40) NOT NULL,
	//     C_NATIONKEY   INTEGER NOT NULL,
	//     C_PHONE       CHAR(15) NOT NULL,
	//     C_ACCTBAL     DECIMAL(15,2)   NOT NULL,
	//     C_MKTSEGMENT  CHAR(10) NOT NULL,
	//     C_COMMENT     VARCHAR(117) NOT NULL
	// );
	auto size = 150000u;

	std::unordered_map<std::int32_t, std::int64_t> primaryMap;

	CUDA::Vector<std::int32_t> custKey(size); // PKey
	CUDA::Vector<std::uint64_t> name(size);
	CUDA::Vector<std::uint64_t> address(size);
	CUDA::Vector<std::int32_t> nationVal(size); // FKey[nation] value
	CUDA::Vector<std::int64_t> nationKey(size); // FKey[nation]
	CUDA::Vector<std::uint64_t> phone(size);
	CUDA::Vector<double> accountBalance(size);
	CUDA::Vector<std::uint64_t> marketSegment(size);
	CUDA::Vector<std::uint64_t> comment(size);

	io::CSVReader<9, io::trim_chars<' ', '\t'>, io::no_quote_escape<'|'>> lineReader(GetTPCHPath("customer.tbl"));
	char *c_custKey, *c_name, *c_address, *c_nationKey, *c_phone, *c_acctBal, *c_mktSegment, *c_comment, *c_end;

	const auto& nationTable = GetTable("nation");
	const auto& nationForeignMap = nationTable->GetPrimaryMap();
	const auto& nationForeignKey = nationTable->GetPrimaryKey();

	auto progress = Utils::Progress::Start("Loading table 'customer'", Utils::Options::Present(Utils::Options::Opt_Print_load));
	auto count = 0u;

	while (lineReader.read_row(
		c_custKey, c_name, c_address, c_nationKey, c_phone, c_acctBal, c_mktSegment, c_comment, c_end
	))
	{
		primaryMap[atoi(c_custKey)] = count;

		custKey[count] = atoi(c_custKey);
		name[count] = StringBucket::HashString(c_name);
		address[count] = StringBucket::HashString(c_address);
		nationVal[count] = atoi(c_nationKey);
		nationKey[count] = nationForeignMap.at(atoi(c_nationKey));
		phone[count] = StringBucket::HashString(c_phone);
		accountBalance[count] = atof(c_acctBal);
		marketSegment[count] = StringBucket::HashString(c_mktSegment);
		comment[count] = StringBucket::HashString(c_comment);

		count++;
		progress.Update(count, size);
	}
	progress.Complete();

	std::vector<std::pair<std::string, ColumnBuffer *>> columns;

	auto primaryKey = new TypedVectorBuffer(new TypedVectorData(new HorseIR::BasicType(HorseIR::BasicType::BasicKind::Int32), std::move(custKey)));
	columns.push_back({"c_custkey", primaryKey});
	columns.push_back({"c_name", new TypedVectorBuffer(new TypedVectorData(new HorseIR::BasicType(HorseIR::BasicType::BasicKind::Symbol), std::move(name)))});
	columns.push_back({"c_address", new TypedVectorBuffer(new TypedVectorData(new HorseIR::BasicType(HorseIR::BasicType::BasicKind::String), std::move(address)))});
	columns.push_back({"c_nationkey", new EnumerationBuffer(nationForeignKey,
		new TypedVectorBuffer(new TypedVectorData(new HorseIR::BasicType(HorseIR::BasicType::BasicKind::Int32), std::move(nationVal))),
		new TypedVectorBuffer(new TypedVectorData(new HorseIR::BasicType(HorseIR::BasicType::BasicKind::Int64), std::move(nationKey)))
	)});
	columns.push_back({"c_phone", new TypedVectorBuffer(new TypedVectorData(new HorseIR::BasicType(HorseIR::BasicType::BasicKind::Symbol), std::move(phone)))});
	columns.push_back({"c_acctbal", new TypedVectorBuffer(new TypedVectorData(new HorseIR::BasicType(HorseIR::BasicType::BasicKind::Float64), std::move(accountBalance)))});
	columns.push_back({"c_mktsegment", new TypedVectorBuffer(new TypedVectorData(new HorseIR::BasicType(HorseIR::BasicType::BasicKind::Symbol), std::move(marketSegment)))});
	columns.push_back({"c_comment", new TypedVectorBuffer(new TypedVectorData(new HorseIR::BasicType(HorseIR::BasicType::BasicKind::String), std::move(comment)))});
	
	auto table = new TableBuffer(columns);
	table->SetPrimaryKey(primaryKey, primaryMap);
	AddTable("tpch", "customer", table);
}

void DataRegistry::LoadTPCHOrderTable()
{
	// CREATE TABLE ORDERS (
	//     O_ORDERKEY       INTEGER NOT NULL,
	//     O_CUSTKEY        INTEGER NOT NULL,
	//     O_ORDERSTATUS    CHAR(1) NOT NULL,
	//     O_TOTALPRICE     DECIMAL(15,2) NOT NULL,
	//     O_ORDERDATE      DATE NOT NULL,
	//     O_ORDERPRIORITY  CHAR(15) NOT NULL,
	//     O_CLERK          CHAR(15) NOT NULL,
	//     O_SHIPPRIORITY   INTEGER NOT NULL,
	//     O_COMMENT        VARCHAR(79) NOT NULL
	// );
	auto size = 1500000u;

	std::unordered_map<std::int32_t, std::int64_t> primaryMap;

	CUDA::Vector<std::int32_t> orderKey(size); // PKey
	CUDA::Vector<std::int32_t> customerVal(size); // FKey[customer] value
	CUDA::Vector<std::int64_t> customerKey(size); // FKey[customer]
	CUDA::Vector<std::int8_t> orderStatus(size);
	CUDA::Vector<double> totalPrice(size);
	CUDA::Vector<std::int32_t> orderDate(size);
	CUDA::Vector<std::uint64_t> orderPriority(size);
	CUDA::Vector<std::uint64_t> clerk(size);
	CUDA::Vector<std::int32_t> shipPriority(size);
	CUDA::Vector<std::uint64_t> comment(size);

	io::CSVReader<10, io::trim_chars<' ', '\t'>, io::no_quote_escape<'|'>> lineReader(GetTPCHPath("orders.tbl"));
	char *o_orderKey, *o_custKey, *o_orderStatus, *o_totalPrice, *o_orderDate, *o_orderPriority, *o_clerk, *o_shipPriority, *o_comment, *o_end;

	const auto& customerTable = GetTable("customer");
	const auto& customerForeignKey = customerTable->GetPrimaryKey();
	const auto& customerForeignMap = customerTable->GetPrimaryMap();

	auto progress = Utils::Progress::Start("Loading table 'orders'", Utils::Options::Present(Utils::Options::Opt_Print_load));
	auto count = 0u;

	while (lineReader.read_row(
		o_orderKey, o_custKey, o_orderStatus, o_totalPrice, o_orderDate, o_orderPriority, o_clerk, o_shipPriority, o_comment, o_end
	))
	{
		primaryMap[atoi(o_orderKey)] = count;

		orderKey[count] = atoi(o_orderKey);
		customerVal[count] = atoi(o_custKey);
		customerKey[count] = customerForeignMap.at(atoi(o_custKey));
		orderStatus[count] = o_orderStatus[0];
		totalPrice[count] = atof(o_totalPrice);
		orderDate[count] = EpochTime(o_orderDate);
		orderPriority[count] = StringBucket::HashString(o_orderPriority);
		clerk[count] = StringBucket::HashString(o_clerk);
		shipPriority[count] = atoi(o_shipPriority);
		comment[count] = StringBucket::HashString(o_comment);

		count++;
		progress.Update(count, size);
	}
	progress.Complete();

	std::vector<std::pair<std::string, ColumnBuffer *>> columns;

	auto primaryKey = new TypedVectorBuffer(new TypedVectorData(new HorseIR::BasicType(HorseIR::BasicType::BasicKind::Int32), std::move(orderKey)));
	columns.push_back({"o_orderkey", primaryKey});
	columns.push_back({"o_custkey", new EnumerationBuffer(customerForeignKey,
		new TypedVectorBuffer(new TypedVectorData(new HorseIR::BasicType(HorseIR::BasicType::BasicKind::Int32), std::move(customerVal))),
		new TypedVectorBuffer(new TypedVectorData(new HorseIR::BasicType(HorseIR::BasicType::BasicKind::Int64), std::move(customerKey)))
	)});
	columns.push_back({"o_orderstatus", new TypedVectorBuffer(new TypedVectorData(new HorseIR::BasicType(HorseIR::BasicType::BasicKind::Char), std::move(orderStatus)))});
	columns.push_back({"o_totalprice", new TypedVectorBuffer(new TypedVectorData(new HorseIR::BasicType(HorseIR::BasicType::BasicKind::Float64), std::move(totalPrice)))});
	columns.push_back({"o_orderdate", new TypedVectorBuffer(new TypedVectorData(new HorseIR::BasicType(HorseIR::BasicType::BasicKind::Date), std::move(orderDate)))});
	columns.push_back({"o_orderpriority", new TypedVectorBuffer(new TypedVectorData(new HorseIR::BasicType(HorseIR::BasicType::BasicKind::Symbol), std::move(orderPriority)))});
	columns.push_back({"o_clerk", new TypedVectorBuffer(new TypedVectorData(new HorseIR::BasicType(HorseIR::BasicType::BasicKind::Symbol), std::move(clerk)))});
	columns.push_back({"o_shippriority", new TypedVectorBuffer(new TypedVectorData(new HorseIR::BasicType(HorseIR::BasicType::BasicKind::Int32), std::move(shipPriority)))});
	columns.push_back({"o_comment", new TypedVectorBuffer(new TypedVectorData(new HorseIR::BasicType(HorseIR::BasicType::BasicKind::String), std::move(comment)))});
	
	auto table = new TableBuffer(columns);
	table->SetPrimaryKey(primaryKey, primaryMap);
	AddTable("tpch", "orders", table);
}

void DataRegistry::LoadTPCHLineItemTable()
{
	// CREATE TABLE LINEITEM (
	//     L_ORDERKEY        INTEGER NOT NULL,
	//     L_PARTKEY         INTEGER NOT NULL,
	//     L_SUPPKEY         INTEGER NOT NULL,
	//     L_LINENUMBER      INTEGER NOT NULL,
	//     L_QUANTITY        DECIMAL(15,2) NOT NULL,
	//     L_EXTENDEDPRICE   DECIMAL(15,2) NOT NULL,
	//     L_DISCOUNT        DECIMAL(15,2) NOT NULL,
	//     L_TAX             DECIMAL(15,2) NOT NULL,
	//     L_RETURNFLAG      CHAR(1) NOT NULL,
	//     L_LINESTATUS      CHAR(1) NOT NULL,
	//     L_SHIPDATE        DATE NOT NULL,
	//     L_COMMITDATE      DATE NOT NULL,
	//     L_RECEIPTDATE     DATE NOT NULL,
	//     L_SHIPINSTRUCT    CHAR(25) NOT NULL,
	//     L_SHIPMODE        CHAR(10) NOT NULL,
	//     L_COMMENT         VARCHAR(44) NOT NULL
	// );
	auto size = 6001215u;

	std::unordered_map<std::int32_t, std::int64_t> primaryKey;

	CUDA::Vector<std::int32_t> orderVal(size); // FKey[orders] value, PKey
	CUDA::Vector<std::int64_t> orderKey(size); // FKey[orders], PKey

	CUDA::Vector<std::int32_t> partVal(size); // FKey[partsupplier], value
	CUDA::Vector<std::int32_t> suppVal(size); // FKey[partsupplier], value
	CUDA::Vector<std::int32_t> lineNumber(size); // PKey

	CUDA::Vector<double> quantity(size);
	CUDA::Vector<double> extPrice(size);
	CUDA::Vector<double> discount(size);
	CUDA::Vector<double> tax(size);

	CUDA::Vector<std::int8_t> returnFlag(size);
	CUDA::Vector<std::int8_t> lineStatus(size);

	CUDA::Vector<std::int32_t> shipDate(size);
	CUDA::Vector<std::int32_t> commitDate(size);
	CUDA::Vector<std::int32_t> receiptDate(size);

	CUDA::Vector<std::uint64_t> shipInstruct(size);
	CUDA::Vector<std::uint64_t> shipMode(size);
	CUDA::Vector<std::uint64_t> comment(size);

	io::CSVReader<17, io::trim_chars<' ', '\t'>, io::no_quote_escape<'|'>> lineReader(GetTPCHPath("lineitem.tbl"));
	char *l_orderKey, *l_partKey, *l_suppKey, *l_lineNumber, *l_quantity, *l_extPrice, *l_discount, *l_tax, *l_returnFlag,
	     *l_lineStatus, *l_shipDate, *l_commitDate, *l_receiptDate, *l_shipInstruct, *l_shipMode, *l_comment, *l_end;

	const auto& orderTable = GetTable("orders");
	const auto& orderForeignKey = orderTable->GetPrimaryKey();
	const auto& orderForeignMap = orderTable->GetPrimaryMap();

	auto progress = Utils::Progress::Start("Loading table 'lineitem'", Utils::Options::Present(Utils::Options::Opt_Print_load));
	auto count = 0u;

	while (lineReader.read_row(
		l_orderKey, l_partKey, l_suppKey, l_lineNumber, l_quantity, l_extPrice, l_discount, l_tax, l_returnFlag,
		l_lineStatus, l_shipDate, l_commitDate, l_receiptDate, l_shipInstruct, l_shipMode, l_comment, l_end
	))
	{
		orderVal[count] = atoi(l_orderKey);
		orderKey[count] = orderForeignMap.at(atoi(l_orderKey));

		//TODO: Foreign key (partsupplier)
		partVal[count] = atoi(l_partKey);
		suppVal[count] = atoi(l_suppKey);
		lineNumber[count] = atoi(l_lineNumber);

		quantity[count] = atof(l_quantity);
		extPrice[count] = atof(l_extPrice);
		discount[count] = atof(l_discount);
		tax[count] = atof(l_tax);

		returnFlag[count] = l_returnFlag[0];
		lineStatus[count] = l_lineStatus[0];

		shipDate[count] = EpochTime(l_shipDate);
		commitDate[count] = EpochTime(l_commitDate);
		receiptDate[count] = EpochTime(l_receiptDate);

		shipInstruct[count] = StringBucket::HashString(l_shipInstruct);
		shipMode[count] = StringBucket::HashString(l_shipMode);
		comment[count] = StringBucket::HashString(l_comment);

		count++;
		progress.Update(count, size);
	}
	progress.Complete();

	std::vector<std::pair<std::string, ColumnBuffer *>> columns;

	columns.push_back({"l_orderkey", new EnumerationBuffer(orderForeignKey,
		new TypedVectorBuffer(new TypedVectorData(new HorseIR::BasicType(HorseIR::BasicType::BasicKind::Int32), std::move(orderVal))),
		new TypedVectorBuffer(new TypedVectorData(new HorseIR::BasicType(HorseIR::BasicType::BasicKind::Int64), std::move(orderKey)))
	)});
	columns.push_back({"l_partkey", new TypedVectorBuffer(new TypedVectorData(new HorseIR::BasicType(HorseIR::BasicType::BasicKind::Int32), std::move(partVal)))});
	columns.push_back({"l_suppkey", new TypedVectorBuffer(new TypedVectorData(new HorseIR::BasicType(HorseIR::BasicType::BasicKind::Int32), std::move(suppVal)))});
	columns.push_back({"l_linenumber", new TypedVectorBuffer(new TypedVectorData(new HorseIR::BasicType(HorseIR::BasicType::BasicKind::Int32), std::move(lineNumber)))});

	columns.push_back({"l_quantity", new TypedVectorBuffer(new TypedVectorData(new HorseIR::BasicType(HorseIR::BasicType::BasicKind::Float64), std::move(quantity)))});
	columns.push_back({"l_extendedprice", new TypedVectorBuffer(new TypedVectorData(new HorseIR::BasicType(HorseIR::BasicType::BasicKind::Float64), std::move(extPrice)))});
	columns.push_back({"l_discount", new TypedVectorBuffer(new TypedVectorData(new HorseIR::BasicType(HorseIR::BasicType::BasicKind::Float64), std::move(discount)))});
	columns.push_back({"l_tax", new TypedVectorBuffer(new TypedVectorData(new HorseIR::BasicType(HorseIR::BasicType::BasicKind::Float64), std::move(tax)))});

	columns.push_back({"l_returnflag", new TypedVectorBuffer(new TypedVectorData(new HorseIR::BasicType(HorseIR::BasicType::BasicKind::Char), std::move(returnFlag)))});
	columns.push_back({"l_linestatus", new TypedVectorBuffer(new TypedVectorData(new HorseIR::BasicType(HorseIR::BasicType::BasicKind::Char), std::move(lineStatus)))});

	columns.push_back({"l_shipdate", new TypedVectorBuffer(new TypedVectorData(new HorseIR::BasicType(HorseIR::BasicType::BasicKind::Date), std::move(shipDate)))});
	columns.push_back({"l_commitdate", new TypedVectorBuffer(new TypedVectorData(new HorseIR::BasicType(HorseIR::BasicType::BasicKind::Date), std::move(commitDate)))});
	columns.push_back({"l_receiptdate", new TypedVectorBuffer(new TypedVectorData(new HorseIR::BasicType(HorseIR::BasicType::BasicKind::Date), std::move(receiptDate)))});
	
	columns.push_back({"l_shipinstruct", new TypedVectorBuffer(new TypedVectorData(new HorseIR::BasicType(HorseIR::BasicType::BasicKind::Symbol), std::move(shipInstruct)))});
	columns.push_back({"l_shipmode", new TypedVectorBuffer(new TypedVectorData(new HorseIR::BasicType(HorseIR::BasicType::BasicKind::Symbol), std::move(shipMode)))});
	columns.push_back({"l_comment", new TypedVectorBuffer(new TypedVectorData(new HorseIR::BasicType(HorseIR::BasicType::BasicKind::String), std::move(comment)))});

	AddTable("tpch", "lineitem", new TableBuffer(columns));
}

void DataRegistry::AddTable(const std::string& db, const std::string& name, TableBuffer *table)
{
	table->SetTag(db + "_" + name);
	m_registry.insert({name, table});
}

TableBuffer *DataRegistry::GetTable(const std::string& name) const
{
	if (m_registry.find(name) == m_registry.end())
	{
		Utils::Logger::LogError("Table '" + name + "' not found");
	}
	return m_registry.at(name);
}

}
