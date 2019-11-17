#include "Runtime/DataRegistry.h"

#include <stdlib.h>

#include "CUDA/Vector.h"

#include "HorseIR/Utils/PrettyPrinter.h"

#include "Libraries/csv.h"

#include "Runtime/DataBuffers/EnumerationBuffer.h"
#include "Runtime/DataBuffers/VectorBuffer.h"

#include "Utils/Date.h"
#include "Utils/Logger.h"

namespace Runtime {

template<typename T>
void DataRegistry::LoadDebugData(std::vector<std::pair<std::string, ColumnBuffer *>>& columns, const HorseIR::BasicType *type, unsigned long size)
{
	CUDA::Vector<T> zeros(size);
	CUDA::Vector<T> ones(size);
	CUDA::Vector<T> asc(size);
	CUDA::Vector<T> desc(size);

	CUDA::Vector<T> fkey(size);
	CUDA::Vector<std::int64_t> indexes(size);
	
	for (auto i = 0u; i < size; ++i)
	{
		zeros[i] = 0;
		ones[i] = 1;
		asc[i] = i;
		desc[i] = size - i - 1;

		fkey[i] = i;
		indexes[i] = (i / 2) * 2;
	}

	auto name = HorseIR::PrettyPrinter::PrettyString(type);
	columns.push_back({"zeros_" + name, new TypedVectorBuffer(new TypedVectorData<T>(type, std::move(zeros)))});
	columns.push_back({"ones_" + name, new TypedVectorBuffer(new TypedVectorData<T>(type, std::move(ones)))});
	columns.push_back({"asc_" + name, new TypedVectorBuffer(new TypedVectorData<T>(type, std::move(asc)))});
	columns.push_back({"desc_" + name, new TypedVectorBuffer(new TypedVectorData<T>(type, std::move(desc)))});

	auto foreignBuffer = new TypedVectorBuffer(new TypedVectorData<T>(type, std::move(fkey)));
	columns.push_back({"fkey_" + name, foreignBuffer});
	columns.push_back({"enum_" + name, new EnumerationBuffer(
		foreignBuffer, new TypedVectorBuffer<std::int64_t>(new TypedVectorData<std::int64_t>(new HorseIR::BasicType(HorseIR::BasicType::BasicKind::Int64), std::move(indexes)))
	)});
}

void DataRegistry::LoadStringDebugData(std::vector<std::pair<std::string, ColumnBuffer *>>& columns, unsigned long size)
{
	CUDA::Vector<std::string> strings(size);
	CUDA::Vector<std::string> symbols(size);

	for (auto i = 0u; i < size; ++i)
	{
		auto numberString = std::to_string(i);
		auto paddedString = std::string(5 - numberString.length(), '0') + numberString;
		strings[i] = "String#" + paddedString;
		symbols[i] = "Symbol#" + paddedString;
	}

	columns.push_back({"strings", new TypedVectorBuffer(new TypedVectorData<std::string>(new HorseIR::BasicType(HorseIR::BasicType::BasicKind::String), std::move(strings)))});
	columns.push_back({"symbols", new TypedVectorBuffer(new TypedVectorData<std::string>(new HorseIR::BasicType(HorseIR::BasicType::BasicKind::Symbol), std::move(symbols)))});
}

void DataRegistry::LoadDebugData()
{
	Utils::Logger::LogSection("Loading debug data");

	for (unsigned long i = 32; i <= 2048; i <<= 1)
	{
		std::vector<std::pair<std::string, ColumnBuffer *>> columns;

		LoadDebugData<int8_t>(columns, new HorseIR::BasicType(HorseIR::BasicType::BasicKind::Boolean), i);
		LoadDebugData<int8_t>(columns, new HorseIR::BasicType(HorseIR::BasicType::BasicKind::Int8), i);
		LoadDebugData<int16_t>(columns, new HorseIR::BasicType(HorseIR::BasicType::BasicKind::Int16), i);
		LoadDebugData<int32_t>(columns, new HorseIR::BasicType(HorseIR::BasicType::BasicKind::Int32), i);
		LoadDebugData<int64_t>(columns, new HorseIR::BasicType(HorseIR::BasicType::BasicKind::Int64), i);
		LoadDebugData<float>(columns, new HorseIR::BasicType(HorseIR::BasicType::BasicKind::Float32), i);
		LoadDebugData<double>(columns, new HorseIR::BasicType(HorseIR::BasicType::BasicKind::Float64), i);
		LoadStringDebugData(columns, i);

		AddTable("debug_" + std::to_string(i), new TableBuffer(columns));
	}
}

void DataRegistry::LoadTPCHData()
{
	Utils::Logger::LogSection("Loading TPC-H data");

	LoadTPCHSupplierTable();
	LoadTPCHPartTable();
	LoadTPCHPartSupplierTable();
	LoadTPCHCustomerTable();
	LoadTPCHOrderTable();
	LoadTPCHLineItemTable();
}

std::int32_t DataRegistry::EpochTime(char *date) const
{
	int year, month, day;
	sscanf(date, "%d-%d-%d", &year, &month, &day);
	return Utils::Date::EpochTime_day(year, month, day);
}

void DataRegistry::LoadTPCHLineItemTable()
{
	std::unordered_map<std::int64_t, std::int64_t> primaryKey;

	CUDA::Vector<std::int64_t> orderKey;
	CUDA::Vector<std::int64_t> partKey;

	CUDA::Vector<double> quantity;
	CUDA::Vector<double> extPrice;
	CUDA::Vector<double> discount;
	CUDA::Vector<double> tax;

	CUDA::Vector<std::int8_t> returnFlag;
	CUDA::Vector<std::int8_t> lineStatus;

	CUDA::Vector<std::int32_t> shipDate;
	CUDA::Vector<std::int32_t> commitDate;
	CUDA::Vector<std::int32_t> receiptDate;

	CUDA::Vector<std::string> shipInstruct;
	CUDA::Vector<std::string> shipMode;

	io::CSVReader<17, io::trim_chars<' ', '\t'>, io::no_quote_escape<'|'>> lineReader("../data/tpc-h/lineitem.tbl");
	char *l_orderKey, *l_partKey, *l_supplyKey, *l_lineNumber, *l_quantity, *l_extPrice, *l_discount, *l_tax, *l_returnFlag,
	     *l_lineStatus, *l_shipDate, *l_commitDate, *l_receiptDate, *l_shipInstruct, *l_shipMode, *l_comment, *l_end;

	const auto& orderTable = GetTable("orders");
	const auto& orderForeignKey = orderTable->GetPrimaryKey();
	const auto& orderForeignMap = orderTable->GetPrimaryMap();

	auto count = 0u;
	while (lineReader.read_row(
		l_orderKey, l_partKey, l_supplyKey, l_lineNumber, l_quantity, l_extPrice, l_discount, l_tax, l_returnFlag,
		l_lineStatus, l_shipDate, l_commitDate, l_receiptDate, l_shipInstruct, l_shipMode, l_comment, l_end
	))
	{

		orderKey.push_back(orderForeignMap.at(atoi(l_orderKey)));
		partKey.push_back(atof(l_partKey));

		quantity.push_back(atof(l_quantity));
		extPrice.push_back(atof(l_extPrice));
		discount.push_back(atof(l_discount));
		tax.push_back(atof(l_tax));

		returnFlag.push_back(l_returnFlag[0]);
		lineStatus.push_back(l_lineStatus[0]);

		shipDate.push_back(EpochTime(l_shipDate));
		commitDate.push_back(EpochTime(l_commitDate));
		receiptDate.push_back(EpochTime(l_receiptDate));

		shipInstruct.push_back(std::string(l_shipInstruct));
		shipMode.push_back(std::string(l_shipMode));

		count++;
	}

	std::vector<std::pair<std::string, ColumnBuffer *>> columns;

	columns.push_back({"l_orderkey", new EnumerationBuffer(
		orderForeignKey, new TypedVectorBuffer<std::int64_t>(new TypedVectorData<std::int64_t>(new HorseIR::BasicType(HorseIR::BasicType::BasicKind::Int64), std::move(orderKey)))
	)});
	columns.push_back({"l_partkey", new TypedVectorBuffer<std::int64_t>(new TypedVectorData<std::int64_t>(new HorseIR::BasicType(HorseIR::BasicType::BasicKind::Int64), std::move(partKey)))});

	columns.push_back({"l_quantity", new TypedVectorBuffer<double>(new TypedVectorData<double>(new HorseIR::BasicType(HorseIR::BasicType::BasicKind::Float64), std::move(quantity)))});
	columns.push_back({"l_extendedprice", new TypedVectorBuffer<double>(new TypedVectorData<double>(new HorseIR::BasicType(HorseIR::BasicType::BasicKind::Float64), std::move(extPrice)))});
	columns.push_back({"l_discount", new TypedVectorBuffer<double>(new TypedVectorData<double>(new HorseIR::BasicType(HorseIR::BasicType::BasicKind::Float64), std::move(discount)))});
	columns.push_back({"l_tax", new TypedVectorBuffer<double>(new TypedVectorData<double>(new HorseIR::BasicType(HorseIR::BasicType::BasicKind::Float64), std::move(tax)))});

	columns.push_back({"l_returnflag", new TypedVectorBuffer<std::int8_t>(new TypedVectorData<std::int8_t>(new HorseIR::BasicType(HorseIR::BasicType::BasicKind::Char), std::move(returnFlag)))});
	columns.push_back({"l_linestatus", new TypedVectorBuffer<std::int8_t>(new TypedVectorData<std::int8_t>(new HorseIR::BasicType(HorseIR::BasicType::BasicKind::Char), std::move(lineStatus)))});

	columns.push_back({"l_shipdate", new TypedVectorBuffer<std::int32_t>(new TypedVectorData<std::int32_t>(new HorseIR::BasicType(HorseIR::BasicType::BasicKind::Date), std::move(shipDate)))});
	columns.push_back({"l_commitdate", new TypedVectorBuffer<std::int32_t>(new TypedVectorData<std::int32_t>(new HorseIR::BasicType(HorseIR::BasicType::BasicKind::Date), std::move(commitDate)))});
	columns.push_back({"l_receiptdate", new TypedVectorBuffer<std::int32_t>(new TypedVectorData<std::int32_t>(new HorseIR::BasicType(HorseIR::BasicType::BasicKind::Date), std::move(receiptDate)))});
	
	columns.push_back({"l_shipinstruct", new TypedVectorBuffer<std::string>(new TypedVectorData<std::string>(new HorseIR::BasicType(HorseIR::BasicType::BasicKind::Symbol), std::move(shipInstruct)))});
	columns.push_back({"l_shipmode", new TypedVectorBuffer<std::string>(new TypedVectorData<std::string>(new HorseIR::BasicType(HorseIR::BasicType::BasicKind::Symbol), std::move(shipMode)))});

	AddTable("lineitem", new TableBuffer(columns));
}

void DataRegistry::LoadTPCHPartTable()
{
	std::unordered_map<std::int64_t, std::int64_t> primaryMap;

	CUDA::Vector<std::int64_t> partKey;
	CUDA::Vector<std::string> brand;
	CUDA::Vector<std::string> type;
	CUDA::Vector<std::int64_t> size;
	CUDA::Vector<std::string> container;

	io::CSVReader<10, io::trim_chars<' ', '\t'>, io::no_quote_escape<'|'>> lineReader("../data/tpc-h/part.tbl");
	char *p_partkey, *p_name, *p_mfgr, *p_brand, *p_type, *p_size, *p_container, *p_retailprice, *p_comment, *p_end;

	auto count = 0u;
	while (lineReader.read_row(
		p_partkey, p_name, p_mfgr, p_brand, p_type, p_size, p_container, p_retailprice, p_comment, p_end
	))
	{
		primaryMap[atoi(p_partkey)] = count;

		partKey.push_back(atoi(p_partkey));
		brand.push_back(std::string(p_brand));
		type.push_back(std::string(p_type));
		size.push_back(atof(p_size));
		container.push_back(std::string(p_container));

		count++;
	}

	std::vector<std::pair<std::string, ColumnBuffer *>> columns;

	auto primaryKey = new TypedVectorBuffer<std::int64_t>(new TypedVectorData<std::int64_t>(new HorseIR::BasicType(HorseIR::BasicType::BasicKind::Int64), std::move(partKey)));
	columns.push_back({"p_partkey", primaryKey});
	columns.push_back({"p_brand", new TypedVectorBuffer<std::string>(new TypedVectorData<std::string>(new HorseIR::BasicType(HorseIR::BasicType::BasicKind::Symbol), std::move(brand)))});
	columns.push_back({"p_type", new TypedVectorBuffer<std::string>(new TypedVectorData<std::string>(new HorseIR::BasicType(HorseIR::BasicType::BasicKind::Symbol), std::move(type)))});
	columns.push_back({"p_size", new TypedVectorBuffer<std::int64_t>(new TypedVectorData<std::int64_t>(new HorseIR::BasicType(HorseIR::BasicType::BasicKind::Int64), std::move(size)))});
	columns.push_back({"p_container", new TypedVectorBuffer<std::string>(new TypedVectorData<std::string>(new HorseIR::BasicType(HorseIR::BasicType::BasicKind::Symbol), std::move(container)))});
	
	auto table = new TableBuffer(columns);
	table->SetPrimaryKey(primaryKey, primaryMap);
	AddTable("part", table);
}

void DataRegistry::LoadTPCHPartSupplierTable()
{
	CUDA::Vector<std::int64_t> partKey;
	CUDA::Vector<std::int64_t> supplierKey;

	io::CSVReader<6, io::trim_chars<' ', '\t'>, io::no_quote_escape<'|'>> lineReader("../data/tpc-h/partsupp.tbl");
	char *ps_partkey, *ps_suppkey, *ps_availqty, *ps_supplycost, *ps_comment, *ps_end;

	const auto& partTable = GetTable("part");
	const auto& supplierTable = GetTable("supplier");

	const auto& partForeignMap = partTable->GetPrimaryMap();
	const auto& partForeignKey = partTable->GetPrimaryKey();

	const auto& supplierForeignMap = supplierTable->GetPrimaryMap();
	const auto& supplierForeignKey = supplierTable->GetPrimaryKey();

	auto count = 0u;
	while (lineReader.read_row(
		ps_partkey, ps_suppkey, ps_availqty, ps_supplycost, ps_comment, ps_end
	))
	{
		partKey.push_back(partForeignMap.at(atoi(ps_partkey)));
		supplierKey.push_back(partForeignMap.at(atoi(ps_suppkey)));

		count++;
	}

	std::vector<std::pair<std::string, ColumnBuffer *>> columns;

	columns.push_back({"ps_partkey", new EnumerationBuffer(
		partForeignKey, new TypedVectorBuffer<std::int64_t>(new TypedVectorData<std::int64_t>(new HorseIR::BasicType(HorseIR::BasicType::BasicKind::Int64), std::move(partKey)))
	)});
	columns.push_back({"ps_suppkey", new EnumerationBuffer(
		supplierForeignKey, new TypedVectorBuffer<std::int64_t>(new TypedVectorData<std::int64_t>(new HorseIR::BasicType(HorseIR::BasicType::BasicKind::Int64), std::move(supplierKey)))
	)});
	
	AddTable("partsupp", new TableBuffer(columns));
}

void DataRegistry::LoadTPCHOrderTable()
{
	std::unordered_map<std::int64_t, std::int64_t> primaryMap;

	CUDA::Vector<std::int64_t> orderKey;
	CUDA::Vector<std::int64_t> customerKey;
	CUDA::Vector<std::int32_t> orderDate;
	CUDA::Vector<std::string> orderPriority;

	io::CSVReader<10, io::trim_chars<' ', '\t'>, io::no_quote_escape<'|'>> lineReader("../data/tpc-h/orders.tbl");
	char *o_orderkey, *o_custkey, *o_orderstatus, *o_totalprice, *o_orderdate, *o_orderpriority, *o_clerk, *o_shippriority, *o_comment, *o_end;

	const auto& customerTable = GetTable("customer");
	const auto& customerForeignKey = customerTable->GetPrimaryKey();
	const auto& customerForeignMap = customerTable->GetPrimaryMap();

	auto count = 0u;
	while (lineReader.read_row(
		o_orderkey, o_custkey, o_orderstatus, o_totalprice, o_orderdate, o_orderpriority, o_clerk, o_shippriority, o_comment, o_end
	))
	{
		primaryMap[atoi(o_orderkey)] = count;

		orderKey.push_back(atoi(o_orderkey));
		customerKey.push_back(customerForeignMap.at(atoi(o_custkey)));
		orderDate.push_back(EpochTime(o_orderdate));
		orderPriority.push_back(std::string(o_orderpriority));

		count++;
	}

	std::vector<std::pair<std::string, ColumnBuffer *>> columns;

	auto primaryKey = new TypedVectorBuffer<std::int64_t>(new TypedVectorData<std::int64_t>(new HorseIR::BasicType(HorseIR::BasicType::BasicKind::Int64), std::move(orderKey)));
	columns.push_back({"o_orderkey", primaryKey});
	columns.push_back({"o_custkey", new EnumerationBuffer(
		customerForeignKey, new TypedVectorBuffer<std::int64_t>(new TypedVectorData<std::int64_t>(new HorseIR::BasicType(HorseIR::BasicType::BasicKind::Int64), std::move(customerKey)))
	)});
	columns.push_back({"o_orderdate", new TypedVectorBuffer<std::int32_t>(new TypedVectorData<std::int32_t>(new HorseIR::BasicType(HorseIR::BasicType::BasicKind::Date), std::move(orderDate)))});
	columns.push_back({"o_orderpriority", new TypedVectorBuffer<std::string>(new TypedVectorData<std::string>(new HorseIR::BasicType(HorseIR::BasicType::BasicKind::Symbol), std::move(orderPriority)))});
	
	auto table = new TableBuffer(columns);
	table->SetPrimaryKey(primaryKey, primaryMap);
	AddTable("orders", table);
}

void DataRegistry::LoadTPCHSupplierTable()
{
	std::unordered_map<std::int64_t, std::int64_t> primaryMap;

	CUDA::Vector<std::int64_t> supplierKey;
	CUDA::Vector<std::string> comment;

	io::CSVReader<8, io::trim_chars<' ', '\t'>, io::no_quote_escape<'|'>> lineReader("../data/tpc-h/supplier.tbl");
	char *s_suppkey, *s_name, *s_address, *s_nationkey, *s_phone, *s_acctbal, *s_comment, *s_end;

	auto count = 0u;
	while (lineReader.read_row(
		s_suppkey, s_name, s_address, s_nationkey, s_phone, s_acctbal, s_comment, s_end
	))
	{
		primaryMap[atoi(s_suppkey)] = count;

		supplierKey.push_back(atoi(s_suppkey));
		comment.push_back(std::string(s_comment));

		count++;
	}

	std::vector<std::pair<std::string, ColumnBuffer *>> columns;

	auto primaryKey = new TypedVectorBuffer<std::int64_t>(new TypedVectorData<std::int64_t>(new HorseIR::BasicType(HorseIR::BasicType::BasicKind::Int64), std::move(supplierKey)));
	columns.push_back({"s_suppkey", primaryKey});
	columns.push_back({"s_comment", new TypedVectorBuffer<std::string>(new TypedVectorData<std::string>(new HorseIR::BasicType(HorseIR::BasicType::BasicKind::Symbol), std::move(comment)))});
	
	auto table = new TableBuffer(columns);
	table->SetPrimaryKey(primaryKey, primaryMap);
	AddTable("supplier", table);
}

void DataRegistry::LoadTPCHCustomerTable()
{
	std::unordered_map<std::int64_t, std::int64_t> primaryMap;

	CUDA::Vector<std::int64_t> custKey;
	CUDA::Vector<std::string> phone;
	CUDA::Vector<double> accountBalance;

	io::CSVReader<9, io::trim_chars<' ', '\t'>, io::no_quote_escape<'|'>> lineReader("../data/tpc-h/customer.tbl");
	char *c_custkey, *c_name, *c_address, *c_nationkey, *c_phone, *c_acctbal, *c_mksegment, *c_comment, *c_end;

	auto count = 0u;
	while (lineReader.read_row(
		c_custkey, c_name, c_address, c_nationkey, c_phone, c_acctbal, c_mksegment, c_comment, c_end
	))
	{
		primaryMap[atoi(c_custkey)] = count;

		custKey.push_back(atoi(c_custkey));
		phone.push_back(std::string(c_phone));
		accountBalance.push_back(atof(c_acctbal));

		count++;
	}

	std::vector<std::pair<std::string, ColumnBuffer *>> columns;

	auto primaryKey = new TypedVectorBuffer<std::int64_t>(new TypedVectorData<std::int64_t>(new HorseIR::BasicType(HorseIR::BasicType::BasicKind::Int64), std::move(custKey)));
	columns.push_back({"c_custkey", primaryKey});
	columns.push_back({"c_phone", new TypedVectorBuffer<std::string>(new TypedVectorData<std::string>(new HorseIR::BasicType(HorseIR::BasicType::BasicKind::String), std::move(phone)))});
	columns.push_back({"c_acctbal", new TypedVectorBuffer<double>(new TypedVectorData<double>(new HorseIR::BasicType(HorseIR::BasicType::BasicKind::Float64), std::move(accountBalance)))});
	
	auto table = new TableBuffer(columns);
	table->SetPrimaryKey(primaryKey, primaryMap);
	AddTable("customer", table);
}

void DataRegistry::AddTable(const std::string& name, TableBuffer *table)
{
	m_registry.insert({name, table});

	Utils::Logger::LogInfo("Loaded table '" + name + "'");
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
