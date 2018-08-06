#include "Runtime/DataRegistry.h"

#include "Utils/Logger.h"

namespace Runtime {

template<typename T>
void DataRegistry::LoadDebugData(DataTable *table, const HorseIR::PrimitiveType *type, unsigned long size)
{
	std::vector<T> zeros(size);
	std::vector<T> ones(size);
	std::vector<T> inc(size);
	
	for (unsigned long i = 0; i < size; ++i)
	{
		zeros.at(i) = 0;
		ones.at(i) = 1;
		inc.at(i) = i;
	}

	table->AddColumn("zeros_" + type->ToString(), new TypedDataVector<T>(type, zeros));
	table->AddColumn("ones_" + type->ToString(), new TypedDataVector<T>(type, ones));
	table->AddColumn("inc_" + type->ToString(), new TypedDataVector<T>(type, inc));
}

void DataRegistry::LoadDebugData()
{
	Utils::Logger::LogSection("Loading debug data");

	for (unsigned long i = 256; i <= 2048; i <<= 1)
	{
		auto table = new DataTable(i);

		LoadDebugData<int8_t>(table, new HorseIR::PrimitiveType(HorseIR::PrimitiveType::Kind::Bool), i);
		LoadDebugData<int8_t>(table, new HorseIR::PrimitiveType(HorseIR::PrimitiveType::Kind::Int8), i);
		LoadDebugData<int16_t>(table, new HorseIR::PrimitiveType(HorseIR::PrimitiveType::Kind::Int16), i);
		LoadDebugData<int32_t>(table, new HorseIR::PrimitiveType(HorseIR::PrimitiveType::Kind::Int32), i);
		LoadDebugData<int64_t>(table, new HorseIR::PrimitiveType(HorseIR::PrimitiveType::Kind::Int64), i);
		LoadDebugData<float>(table, new HorseIR::PrimitiveType(HorseIR::PrimitiveType::Kind::Float32), i);
		LoadDebugData<double>(table, new HorseIR::PrimitiveType(HorseIR::PrimitiveType::Kind::Float64), i);

		AddTable("debug_" + std::to_string(i), table);
	}
}

void DataRegistry::AddTable(const std::string& name, DataTable *table)
{
	m_registry.insert({name, table});

	Utils::Logger::LogInfo("Loaded table '" + name + "'");
}

DataTable *DataRegistry::GetTable(const std::string& name) const
{
	if (m_registry.find(name) == m_registry.end())
	{
		Utils::Logger::LogError("Table '" + name + "' not found");
	}
	return m_registry.at(name);
}

void LoadFile(const std::string& name)
{
	//TODO: Load data from file into the registry
}

}
