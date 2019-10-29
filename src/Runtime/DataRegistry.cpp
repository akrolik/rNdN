#include "Runtime/DataRegistry.h"

#include <stdlib.h>

#include "CUDA/Vector.h"

#include "HorseIR/Utils/PrettyPrinter.h"

#include "Libraries/csv.h"

#include "Utils/Logger.h"

namespace Runtime {

template<typename T>
void DataRegistry::LoadDebugData(std::vector<std::pair<std::string, VectorBuffer *>>& columns, const HorseIR::BasicType *type, unsigned long size)
{
	CUDA::Vector<T> zeros(size);
	CUDA::Vector<T> ones(size);
	CUDA::Vector<T> asc(size);
	CUDA::Vector<T> desc(size);
	
	for (unsigned long i = 0; i < size; ++i)
	{
		zeros.at(i) = 0;
		ones.at(i) = 1;
		asc.at(i) = i;
		desc.at(i) = size - i - 1;
	}

	auto name = HorseIR::PrettyPrinter::PrettyString(type);
	columns.push_back({"zeros_" + name, new TypedVectorBuffer(new TypedVectorData<T>(type, zeros))});
	columns.push_back({"ones_" + name, new TypedVectorBuffer(new TypedVectorData<T>(type, ones))});
	columns.push_back({"asc_" + name, new TypedVectorBuffer(new TypedVectorData<T>(type, asc))});
	columns.push_back({"desc_" + name, new TypedVectorBuffer(new TypedVectorData<T>(type, desc))});
}

void DataRegistry::LoadDebugData()
{
	Utils::Logger::LogSection("Loading debug data");

	for (unsigned long i = 256; i <= 2048; i <<= 1)
	{
		std::vector<std::pair<std::string, VectorBuffer *>> columns;

		LoadDebugData<int8_t>(columns, new HorseIR::BasicType(HorseIR::BasicType::BasicKind::Boolean), i);
		LoadDebugData<int8_t>(columns, new HorseIR::BasicType(HorseIR::BasicType::BasicKind::Int8), i);
		LoadDebugData<int16_t>(columns, new HorseIR::BasicType(HorseIR::BasicType::BasicKind::Int16), i);
		LoadDebugData<int32_t>(columns, new HorseIR::BasicType(HorseIR::BasicType::BasicKind::Int32), i);
		LoadDebugData<int64_t>(columns, new HorseIR::BasicType(HorseIR::BasicType::BasicKind::Int64), i);
		LoadDebugData<float>(columns, new HorseIR::BasicType(HorseIR::BasicType::BasicKind::Float32), i);
		LoadDebugData<double>(columns, new HorseIR::BasicType(HorseIR::BasicType::BasicKind::Float64), i);

		AddTable("debug_" + std::to_string(i), new TableBuffer(columns));
	}
}

void DataRegistry::LoadTPCHData()
{
	Utils::Logger::LogSection("Loading TPC-H data");

	CUDA::Vector<double> quantity;
	CUDA::Vector<double> extPrice;
	CUDA::Vector<double> discount;
	CUDA::Vector<std::int32_t> shipDate;

	io::CSVReader<17, io::trim_chars<' ', '\t'>, io::no_quote_escape<'|'>> reader("../data/tpc-h/lineitem.tbl");

	char *l_orderKey, *l_partKey, *l_supplyKey, *l_lineNumber, *l_quantity, *l_extPrice, *l_discount, *l_tax, *l_returnFlag,
	     *l_lineStatus, *l_shipDate, *l_commitDate, *l_receiptDate, *l_shipInstruct, *l_shipMode, *l_comment, *l_end;

	auto count = 0u;
	while (reader.read_row(
		l_orderKey, l_partKey, l_supplyKey, l_lineNumber, l_quantity, l_extPrice, l_discount, l_tax, l_returnFlag,
		l_lineStatus, l_shipDate, l_commitDate, l_receiptDate, l_shipInstruct, l_shipMode, l_comment, l_end
	))
	{
		quantity.push_back(atof(l_quantity));
		extPrice.push_back(atof(l_extPrice));
		discount.push_back(atof(l_discount));

		int year, month, day;
		sscanf(l_shipDate, "%d-%d-%d", &year, &month, &day);
		auto date_val = new HorseIR::DateValue(year, month, day);
		shipDate.push_back(date_val->GetEpochTime());
		delete date_val;

		count++;
	}

	std::vector<std::pair<std::string, VectorBuffer *>> columns;

	columns.push_back({"l_quantity", new TypedVectorBuffer<double>(new TypedVectorData<double>(new HorseIR::BasicType(HorseIR::BasicType::BasicKind::Float64), quantity))});
	columns.push_back({"l_extendedprice", new TypedVectorBuffer<double>(new TypedVectorData<double>(new HorseIR::BasicType(HorseIR::BasicType::BasicKind::Float64), extPrice))});
	columns.push_back({"l_discount", new TypedVectorBuffer<double>(new TypedVectorData<double>(new HorseIR::BasicType(HorseIR::BasicType::BasicKind::Float64), discount))});
	columns.push_back({"l_shipdate", new TypedVectorBuffer<std::int32_t>(new TypedVectorData<std::int32_t>(new HorseIR::BasicType(HorseIR::BasicType::BasicKind::Date), shipDate))});
	
	AddTable("lineitem", new TableBuffer(columns));
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
