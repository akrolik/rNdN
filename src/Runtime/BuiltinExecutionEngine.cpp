#include "Runtime/BuiltinExecutionEngine.h"

#include <numeric>
#include <algorithm>

#include "HorseIR/Utils/TypeUtils.h"

#include "Libraries/jpcre2.hpp"

#include "Runtime/DataBuffers/BufferUtils.h"
#include "Runtime/DataBuffers/ColumnBuffer.h"
#include "Runtime/DataBuffers/DictionaryBuffer.h"
#include "Runtime/DataBuffers/EnumerationBuffer.h"
#include "Runtime/DataBuffers/ListBuffer.h"
#include "Runtime/DataBuffers/TableBuffer.h"
#include "Runtime/DataBuffers/VectorBuffer.h"

#include "Runtime/GPU/Library/GPUSortEngine.h"
#include "Runtime/GPU/Library/GPUGroupEngine.h"
#include "Runtime/GPU/Library/GPUUniqueEngine.h"
#include "Runtime/GPU/Library/GPUJoinEngine.h"

#include "Utils/Logger.h"

namespace Runtime {

std::vector<DataBuffer *> BuiltinExecutionEngine::Execute(const HorseIR::BuiltinFunction *function, const std::vector<DataBuffer *>& arguments)
{
	Utils::Logger::LogDebug("Executing builtin function '" + function->GetName() + "'");

#define Error(m) Utils::Logger::LogError("Builtin function '" + function->GetName() + "' " + m);

	switch (function->GetPrimitive())
	{
		// Algebraic Unary
		case HorseIR::BuiltinFunction::Primitive::Order:
		{
			// Collect the columns for the sort - decomposing the list into individual vectors

			std::vector<VectorBuffer *> sortBuffers;
			if (auto listSort = BufferUtils::GetBuffer<ListBuffer>(arguments.at(0), false))
			{
				for (auto buffer : listSort->GetCells())
				{
					sortBuffers.push_back(BufferUtils::GetBuffer<VectorBuffer>(buffer));
				}
			}
			else if (auto vectorSort = BufferUtils::GetBuffer<VectorBuffer>(arguments.at(0), false))
			{
				sortBuffers.push_back(vectorSort);
			}
			else
			{
				Error("expects either a single vector or a list of vectors");
			}

			// Get a CPU vector or the orders

			auto& orders = BufferUtils::GetVectorBuffer<std::int8_t>(arguments.at(1))->GetCPUReadBuffer()->GetValues();
			if (orders.size() != sortBuffers.size())
			{
				Error("expects equal number of columns as direction specifiers [" + std::to_string(sortBuffers.size()) + " != " + std::to_string(orders.size()) + "]");
			}
			std::vector<std::int8_t> _orders(std::begin(orders), std::end(orders));

			// Sort!

			auto sortSize = 0;
			std::vector<const VectorData *> sortData;

			// Collect sort size and CPU data objects

			bool first = true;
			for (auto buffer : sortBuffers)
			{
				auto bufferSize = buffer->GetElementCount();
				if (first)
				{
					sortSize = bufferSize;
					first = false;
				}
				else if (sortSize != bufferSize)
				{
					Utils::Logger::LogError("Sort requires all columns have equal size [" + std::to_string(sortSize) + " != " + std::to_string(bufferSize) + "]");
				}
				sortData.push_back(buffer->GetCPUReadBuffer());
			}

			CUDA::Vector<std::int64_t> indexes(sortSize);
			std::iota(indexes.begin(), indexes.end(), 0);

			// Sort indexes using the values in sort buffers

			std::sort(indexes.begin(), indexes.end(), [&sortData,&orders](std::int64_t i1, std::int64_t i2)
			{
				// Return true if first element ordered before the second

				auto dataIndex = 0;
				for (const auto data : sortData)
				{
					if (!data->IsEqual(i1, i2))
					{
						return (data->IsSorted(i1, i2) == orders.at(dataIndex));
					}
					dataIndex++;
				}
				return false;
			});

			auto indexData = new TypedVectorData<std::int64_t>(new HorseIR::BasicType(HorseIR::BasicType::BasicKind::Int64), std::move(indexes));
			return {new TypedVectorBuffer<std::int64_t>(indexData)};
		}
		case HorseIR::BuiltinFunction::Primitive::GPUOrderLib:
		{
			// GPU sort!

			GPUSortEngine sortEngine(m_runtime, m_program);
			auto [indexBuffer, dataBuffer] = sortEngine.Sort(arguments);

			// Data buffers can be deallocated as they are unused in a simple sort

			delete dataBuffer;
			return {indexBuffer};
		}
		case HorseIR::BuiltinFunction::Primitive::GPUGroupLib:
		{
			// GPU group!

			GPUGroupEngine groupEngine(m_runtime, m_program);
			return {groupEngine.Group(arguments)};
		}
		case HorseIR::BuiltinFunction::Primitive::GPUUniqueLib:
		{
			// GPU unique!

			GPUUniqueEngine uniqueEngine(m_runtime, m_program);
			return {uniqueEngine.Unique(arguments)};
		}
		case HorseIR::BuiltinFunction::Primitive::GPUJoinLib:
		{
			// GPU join!

			GPUJoinEngine joinEngine(m_runtime, m_program);
			return {joinEngine.Join(arguments)};
		}

		// List
		case HorseIR::BuiltinFunction::Primitive::List:
		{
			return {new ListBuffer(arguments)};
		}

		// Database
		case HorseIR::BuiltinFunction::Primitive::Table:
		{
			auto columnNames = BufferUtils::GetVectorBuffer<std::uint64_t>(arguments.at(0))->GetCPUReadBuffer();
			auto columnValues = BufferUtils::GetBuffer<ListBuffer>(arguments.at(1));

			if (columnNames->GetElementCount() != columnValues->GetCellCount())
			{
				Error("expects header and columns of same size [" + std::to_string(columnNames->GetElementCount()) + " != " + std::to_string(columnValues->GetCellCount()) + "]");
			}

			auto i = 0u;
			std::vector<std::pair<std::string, ColumnBuffer *>> columns;
			for (const auto& columnHash : columnNames->GetValues())
			{
				const auto& columnName = StringBucket::RecoverString(columnHash);
				columns.push_back({columnName, BufferUtils::GetBuffer<VectorBuffer>(columnValues->GetCell(i++))});
			}
			return {new TableBuffer(columns)};
		}
		case HorseIR::BuiltinFunction::Primitive::Keys:
		{
			auto argument = arguments.at(0);
			if (auto dictionary = BufferUtils::GetBuffer<DictionaryBuffer>(argument, false))
			{
				return {dictionary->GetKeys()};
			}
			else if (auto enumeration = BufferUtils::GetBuffer<EnumerationBuffer>(argument, false))
			{
				return {enumeration->GetKeys()};
			}
			Error("unsupported target type " + HorseIR::TypeUtils::TypeString(argument->GetType()));
		}
		case HorseIR::BuiltinFunction::Primitive::Values:
		{
			auto argument = arguments.at(0);
			if (auto dictionary = BufferUtils::GetBuffer<DictionaryBuffer>(argument, false))
			{
				return {dictionary->GetValues()};
			}
			else if (auto enumeration = BufferUtils::GetBuffer<EnumerationBuffer>(argument, false))
			{
				return {enumeration->GetIndexes()};
			}
			Error("unsupported target type " + HorseIR::TypeUtils::TypeString(argument->GetType()));
		}
		case HorseIR::BuiltinFunction::Primitive::Fetch:
		{
			auto argument = arguments.at(0);
			auto enumeration = BufferUtils::GetBuffer<EnumerationBuffer>(argument);
			return {enumeration->GetValues()};
		}
		case HorseIR::BuiltinFunction::Primitive::ColumnValue:
		{
			auto table = BufferUtils::GetBuffer<TableBuffer>(arguments.at(0));
			auto columnSymbol = BufferUtils::GetVectorBuffer<std::uint64_t>(arguments.at(1))->GetCPUReadBuffer();

			if (columnSymbol->GetElementCount() != 1)
			{
				Error("expects a single column argument, received " + std::to_string(columnSymbol->GetElementCount()));
			}

			const auto& columnName = StringBucket::RecoverString(columnSymbol->GetValue(0));
			return {table->GetColumn(columnName)};
		}
		case HorseIR::BuiltinFunction::Primitive::LoadTable:
		{
			auto& dataRegistry = m_runtime.GetDataRegistry();
			auto tableSymbol = BufferUtils::GetVectorBuffer<std::uint64_t>(arguments.at(0))->GetCPUReadBuffer();

			if (tableSymbol->GetElementCount() != 1)
			{
				Error("expects a single table argument, received " + std::to_string(tableSymbol->GetElementCount()));
			}

			const auto& tableName = StringBucket::RecoverString(tableSymbol->GetValue(0));
			return {dataRegistry.GetTable(tableName)};
		}

		// Indexing
		case HorseIR::BuiltinFunction::Primitive::Index:
		{
			// Fetch from the list buffer at a particular index

			auto listBuffer = BufferUtils::GetBuffer<ListBuffer>(arguments.at(0));
			auto indexBuffer = BufferUtils::GetVectorBuffer<std::int64_t>(arguments.at(1));

			auto indexData = indexBuffer->GetCPUReadBuffer();
			auto index = indexData->GetValue(0);

			return {listBuffer->GetCell(index)};
		}

		// Other
		case HorseIR::BuiltinFunction::Primitive::Like:
		{
			auto& stringData = BufferUtils::GetVectorBuffer<std::uint64_t>(arguments.at(0))->GetCPUReadBuffer()->GetValues();
			auto patternData = BufferUtils::GetVectorBuffer<std::uint64_t>(arguments.at(1))->GetCPUReadBuffer();

			if (patternData->GetElementCount() != 1)
			{
				Error("expects a single pattern argument, received " + std::to_string(patternData->GetElementCount()));
			}

			// Transform from SQL like to regex
			//  - Escape: '.', '*', and '\'
			//  - Replace: '%' by '.*' (0 or more) and '_' by '.' (exactly 1)

			const auto& likePatternString = StringBucket::RecoverString(patternData->GetValue(0));
			const auto likePatternSize = likePatternString.size();

			const char *likePattern = likePatternString.c_str();
			char *regexPattern = (char *)malloc(sizeof(char) * likePatternSize * 2 + 2);

			auto j = 0u;
			for (auto i = 0u; i < likePatternSize; ++i)
			{
				char c = likePattern[i];
				if (c == '.' || c == '*' || c == '\\')
				{
					regexPattern[j++] = '\\';
					regexPattern[j++] = c;
				}
				else if (c == '%')
				{
					regexPattern[j++] = '.';
					regexPattern[j++] = '*';
				}
				else if (c == '_')
				{
					regexPattern[j++] = '.';
				}
				else
				{
					regexPattern[j++] = c;
				}
			}
			regexPattern[j++] = '$';
			regexPattern[j] = '\0';
			std::string regexPatternString(regexPattern);

			// Compile the regex and match on all data strings

			jpcre2::select<char>::Regex regex(regexPatternString, PCRE2_ANCHORED, jpcre2::JIT_COMPILE);
			if (!regex)
			{
				Error("unable to compile regex pattern '" + regexPatternString + "' from like pattern '" + likePatternString + "'");
			}

			const auto size = stringData.size();
			CUDA::Vector<std::int8_t> likeData(size);

			for (auto i = 0u; i < size; ++i)
			{
				likeData.at(i) = regex.match(StringBucket::RecoverString(stringData.at(i)));
			}

			return {new TypedVectorBuffer(new TypedVectorData<std::int8_t>(new HorseIR::BasicType(HorseIR::BasicType::BasicKind::Boolean), std::move(likeData)))};
		}
		case HorseIR::BuiltinFunction::Primitive::SubString:
		{
			const auto& stringData = BufferUtils::GetVectorBuffer<std::uint64_t>(arguments.at(0))->GetCPUReadBuffer()->GetValues();
			auto rangeVector = BufferUtils::GetBuffer<VectorBuffer>(arguments.at(1));

			if (rangeVector->GetElementCount() != 2)
			{
				Error("expects 2 element range vector, received " + std::to_string(rangeVector->GetElementCount()));
			}

			size_t position = 0;
			size_t length = 0;
			switch (rangeVector->GetType()->GetBasicKind())
			{
				case HorseIR::BasicType::BasicKind::Boolean:
				case HorseIR::BasicType::BasicKind::Char:
				case HorseIR::BasicType::BasicKind::Int8:
				{
					auto range = BufferUtils::GetVectorBuffer<std::int8_t>(rangeVector)->GetCPUReadBuffer();
					position = range->GetValue(0) - 1;
					length = range->GetValue(1);
					break;
				}
				case HorseIR::BasicType::BasicKind::Int16:
				{
					auto range = BufferUtils::GetVectorBuffer<std::int16_t>(rangeVector)->GetCPUReadBuffer();
					position = range->GetValue(0) - 1;
					length = range->GetValue(1);
					break;
				}
				case HorseIR::BasicType::BasicKind::Int32:
				{
					auto range = BufferUtils::GetVectorBuffer<std::int32_t>(rangeVector)->GetCPUReadBuffer();
					position = range->GetValue(0) - 1;
					length = range->GetValue(1);
					break;
				}
				case HorseIR::BasicType::BasicKind::Int64:
				{
					auto range = BufferUtils::GetVectorBuffer<std::int64_t>(rangeVector)->GetCPUReadBuffer();
					position = range->GetValue(0) - 1;
					length = range->GetValue(1);
					break;
				}
				default:
				{
					Error("range type " + HorseIR::TypeUtils::TypeString(rangeVector->GetType()) + " not supported");
				}
			}

			const auto size = stringData.size();
			CUDA::Vector<std::uint64_t> substringData(size);

			for (auto i = 0u; i < size; ++i)
			{
				substringData[i] = StringBucket::HashString(StringBucket::RecoverString(stringData[i]).substr(position, length));
			}

			return {new TypedVectorBuffer(new TypedVectorData<std::uint64_t>(new HorseIR::BasicType(HorseIR::BasicType::BasicKind::String), std::move(substringData)))};
		}
		default:
		{
			Error("not implemented");
		}
	}
}

}
