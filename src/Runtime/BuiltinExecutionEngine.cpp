#include "Runtime/BuiltinExecutionEngine.h"

#include "HorseIR/Utils/TypeUtils.h"

#include "Runtime/DataBuffers/ListBuffer.h"
#include "Runtime/DataBuffers/TableBuffer.h"
#include "Runtime/DataBuffers/VectorBuffer.h"

#include "Utils/Logger.h"

namespace Runtime {

std::vector<DataBuffer *> BuiltinExecutionEngine::Execute(const HorseIR::BuiltinFunction *function, const std::vector<DataBuffer *>& arguments)
{
	Utils::Logger::LogInfo("Executing builtin function '" + function->GetName() + "'");

	//TODO: We need to verify all casts before they execute
	switch (function->GetPrimitive())
	{
		case HorseIR::BuiltinFunction::Primitive::List:
		{
			return {new ListBuffer(arguments)};
		}
		case HorseIR::BuiltinFunction::Primitive::Table:
		{
			auto columnNames = static_cast<TypedVectorBuffer<HorseIR::SymbolValue *> *>(arguments.at(0))->GetCPUReadBuffer();
			auto columnValues = static_cast<ListBuffer *>(arguments.at(1));

			//TODO: Ensure the columnValues have tabular size
			//TODO: Check the column names have the same size as the columns
			//TODO: Determine the correct table size
			auto table = new TableBuffer(1);

			auto i = 0u;
			for (const auto& columnName : columnNames->GetValues())
			{
				auto columnVector = static_cast<VectorBuffer *>(columnValues->GetCell(i++));
				table->AddColumn(columnName->GetName(), columnVector);
			}

			return {table};
		}
		case HorseIR::BuiltinFunction::Primitive::ColumnValue:
		{
			auto table = static_cast<TableBuffer *>(arguments.at(0));
			auto columnSymbol = static_cast<TypedVectorBuffer<HorseIR::SymbolValue *> *>(arguments.at(1))->GetCPUReadBuffer();

			if (columnSymbol->GetElementCount() != 1)
			{
				Utils::Logger::LogError("Builtin function '" + function->GetName() + "' expects a single column argument");
			}

			return {table->GetColumn(columnSymbol->GetValue(0)->GetName())};
		}
		case HorseIR::BuiltinFunction::Primitive::LoadTable:
		{
			auto& dataRegistry = m_runtime.GetDataRegistry();
			auto tableSymbol = static_cast<TypedVectorBuffer<HorseIR::SymbolValue *> *>(arguments.at(0))->GetCPUReadBuffer();

			if (tableSymbol->GetElementCount() != 1)
			{
				Utils::Logger::LogError("Builtin function '" + function->GetName() + "' expects a single table argument");
			}

			return {dataRegistry.GetTable(tableSymbol->GetValue(0)->GetName())};
		}
		default:
		{
			Utils::Logger::LogError("Builtin function '" + function->GetName() + "' not implemented");
		}
	}
}

}
