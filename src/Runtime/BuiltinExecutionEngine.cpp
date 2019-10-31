#include "Runtime/BuiltinExecutionEngine.h"

#include "HorseIR/Utils/TypeUtils.h"

#include "Runtime/DataBuffers/BufferUtils.h"
#include "Runtime/DataBuffers/ListBuffer.h"
#include "Runtime/DataBuffers/TableBuffer.h"
#include "Runtime/DataBuffers/VectorBuffer.h"

#include "Runtime/GPULibrary/GPUSortEngine.h"
#include "Runtime/GPULibrary/GPUGroupEngine.h"

#include "Utils/Logger.h"

namespace Runtime {

std::vector<DataBuffer *> BuiltinExecutionEngine::Execute(const HorseIR::BuiltinFunction *function, const std::vector<DataBuffer *>& arguments)
{
	Utils::Logger::LogInfo("Executing builtin function '" + function->GetName() + "'");

#define Error(m) Utils::Logger::LogError("Builtin function '" + function->GetName() + "' " + m);

	switch (function->GetPrimitive())
	{
		case HorseIR::BuiltinFunction::Primitive::Order:
		{
			// Collect the columns for the sort - decomposing the list into individual vectors

			std::vector<VectorBuffer *> columns;
			if (auto listSort = BufferUtils::GetBuffer<ListBuffer>(arguments.at(0), false))
			{
				for (auto buffer : listSort->GetCells())
				{
					columns.push_back(BufferUtils::GetBuffer<VectorBuffer>(buffer));
				}
			}
			else if (auto vectorSort = BufferUtils::GetBuffer<VectorBuffer>(arguments.at(0), false))
			{
				columns.push_back(vectorSort);
			}
			else
			{
				Error("expects either a single vector or a list of vectors");
			}

			// Get a CPU vector or the orders

			auto orders = BufferUtils::GetVectorBuffer<char>(arguments.at(1))->GetCPUReadBuffer()->GetValues();
			if (orders.size() != columns.size())
			{
				Error("expects equal number of columns as direction specifiers [" + std::to_string(columns.size()) + " != " + std::to_string(orders.size()) + "]");
			}
			std::vector<char> _orders(std::begin(orders), std::end(orders));

			// Sort!

			GPUSortEngine sortEngine(m_runtime);
			auto [indexBuffer, dataBuffers] = sortEngine.Sort(columns, _orders);

			// Free the sort buffers

			for (auto dataBuffer : dataBuffers)
			{
				delete dataBuffer->GetGPUWriteBuffer();
			}

			return {indexBuffer};
		}
		case HorseIR::BuiltinFunction::Primitive::Group:
		{
			// Collect the columns for the group - decomposing the list into individual vectors

			std::vector<VectorBuffer *> columns;
			if (auto listSort = BufferUtils::GetBuffer<ListBuffer>(arguments.at(0), false))
			{
				for (auto buffer : listSort->GetCells())
				{
					columns.push_back(BufferUtils::GetBuffer<VectorBuffer>(buffer));
				}
			}
			else if (auto vectorSort = BufferUtils::GetBuffer<VectorBuffer>(arguments.at(0), false))
			{
				columns.push_back(vectorSort);
			}
			else
			{
				Error("expects either a single vector or a list of vectors");
			}

			// Group!

			GPUGroupEngine groupEngine(m_runtime);
			return {groupEngine.Group(columns)};
		}
		case HorseIR::BuiltinFunction::Primitive::List:
		{
			return {new ListBuffer(arguments)};
		}
		case HorseIR::BuiltinFunction::Primitive::Table:
		{
			auto columnNames = BufferUtils::GetVectorBuffer<HorseIR::SymbolValue *>(arguments.at(0))->GetCPUReadBuffer();
			auto columnValues = BufferUtils::GetBuffer<ListBuffer>(arguments.at(1));

			if (columnNames->GetElementCount() != columnValues->GetCellCount())
			{
				Error("expects header and columns of same size [" + std::to_string(columnNames->GetElementCount()) + " != " + std::to_string(columnValues->GetCellCount()) + "]");
			}

			auto i = 0u;
			std::vector<std::pair<std::string, VectorBuffer *>> columns;
			for (const auto& columnName : columnNames->GetValues())
			{
				columns.push_back({columnName->GetName(), BufferUtils::GetBuffer<VectorBuffer>(columnValues->GetCell(i++))});
			}
			return {new TableBuffer(columns)};
		}
		case HorseIR::BuiltinFunction::Primitive::ColumnValue:
		{
			auto table = BufferUtils::GetBuffer<TableBuffer>(arguments.at(0));
			auto columnSymbol = BufferUtils::GetVectorBuffer<HorseIR::SymbolValue *>(arguments.at(1))->GetCPUReadBuffer();

			if (columnSymbol->GetElementCount() != 1)
			{
				Error("expects a single column argument");
			}

			return {table->GetColumn(columnSymbol->GetValue(0)->GetName())};
		}
		case HorseIR::BuiltinFunction::Primitive::LoadTable:
		{
			auto& dataRegistry = m_runtime.GetDataRegistry();
			auto tableSymbol = BufferUtils::GetVectorBuffer<HorseIR::SymbolValue *>(arguments.at(0))->GetCPUReadBuffer();

			if (tableSymbol->GetElementCount() != 1)
			{
				Error("expects a single table argument");
			}

			return {dataRegistry.GetTable(tableSymbol->GetValue(0)->GetName())};
		}
		default:
		{
			Error("not implemented");
		}
	}
}

}
