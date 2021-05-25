#include "Runtime/BuiltinExecutionEngine.h"

#include <numeric>
#include <algorithm>

#include "HorseIR/Utils/TypeUtils.h"

#include "Libraries/jpcre2.hpp"

#include "Runtime/DataBuffers/BufferUtils.h"
#include "Runtime/DataBuffers/ColumnBuffer.h"
#include "Runtime/DataBuffers/DictionaryBuffer.h"
#include "Runtime/DataBuffers/EnumerationBuffer.h"
#include "Runtime/DataBuffers/KeyedTableBuffer.h"
#include "Runtime/DataBuffers/ListBuffer.h"
#include "Runtime/DataBuffers/ListCellBuffer.h"
#include "Runtime/DataBuffers/TableBuffer.h"
#include "Runtime/DataBuffers/VectorBuffer.h"

#include "Runtime/GPU/Library/GroupEngine.h"
#include "Runtime/GPU/Library/HashJoinEngine.h"
#include "Runtime/GPU/Library/LoopJoinEngine.h"
#include "Runtime/GPU/Library/SortEngine.h"
#include "Runtime/GPU/Library/UniqueEngine.h"

#include "Utils/Logger.h"
#include "Utils/String.h"

namespace Runtime {

std::vector<DataBuffer *> BuiltinExecutionEngine::Execute(const HorseIR::BuiltinFunction *function, const std::vector<DataBuffer *>& arguments)
{
	if (Utils::Options::IsDebug_Print())
	{
		Utils::Logger::LogDebug("Executing builtin function '" + function->GetName() + "'");
	}

#define Error(m) Utils::Logger::LogError("Builtin function '" + function->GetName() + "' " + m);

	switch (function->GetPrimitive())
	{
		// Algebraic Unary
		case HorseIR::BuiltinFunction::Primitive::Flip:
		{
			//TODO: @flip
			Error("unimplemented");
		}
		case HorseIR::BuiltinFunction::Primitive::Order:
		{
			// Collect the columns for the sort - decomposing the list into individual vectors

			std::vector<const VectorBuffer *> sortBuffers;
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

		// Algebraic binary
		case HorseIR::BuiltinFunction::Primitive::Append:
		{
			//TODO: @append enum/list
			Error("unimplemented");
		}
		case HorseIR::BuiltinFunction::Primitive::Replicate:
		{
			//TODO: @replicate
			Error("unimplemented");
		}

		// List
		case HorseIR::BuiltinFunction::Primitive::List:
		{
			return {new ListCellBuffer(arguments)};
		}

		// Database
		case HorseIR::BuiltinFunction::Primitive::Enum:
		{
			//TODO: @enum
			Error("unimplemented");	
		}
		case HorseIR::BuiltinFunction::Primitive::Dictionary:
		{
			//TODO: @dict
			Error("unimplemented");	
		}
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
		case HorseIR::BuiltinFunction::Primitive::KeyedTable:
		{
			auto key = BufferUtils::GetBuffer<TableBuffer>(arguments.at(0));
			auto value = BufferUtils::GetBuffer<TableBuffer>(arguments.at(1));
			return {new KeyedTableBuffer(key, value)};
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
			else if (auto table = BufferUtils::GetBuffer<TableBuffer>(argument, false))
			{
				// Vector of column names
				
				CUDA::Vector<std::uint64_t> names;
				for (const auto& [name, _] : table->GetColumns())
				{
					names.push_back(StringBucket::HashString(name));
				}
				return {new TypedVectorBuffer(new TypedVectorData(new HorseIR::BasicType(HorseIR::BasicType::BasicKind::Symbol), std::move(names)))};
			}
			else if (auto keyedTable = BufferUtils::GetBuffer<KeyedTableBuffer>(argument, false))
			{
				return {keyedTable->GetKey()};
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
			else if (auto table = BufferUtils::GetBuffer<TableBuffer>(argument, false))
			{
				// List of column data

				std::vector<DataBuffer *> cells;
				for (const auto& [_, column] : table->GetColumns())
				{
					cells.push_back(column);
				}
				return {new ListCellBuffer(cells)};
			}
			else if (auto keyedTable = BufferUtils::GetBuffer<KeyedTableBuffer>(argument, false))
			{
				return {keyedTable->GetValue()};
			}
			Error("unsupported target type " + HorseIR::TypeUtils::TypeString(argument->GetType()));
		}
		case HorseIR::BuiltinFunction::Primitive::Meta:
		{
			std::vector<std::pair<std::string, ColumnBuffer *>> columns;
			CUDA::Vector<std::uint64_t> names;
			CUDA::Vector<std::uint64_t> types;
			CUDA::Vector<std::uint64_t> attributes;

			auto argument = arguments.at(0);
			if (auto table = BufferUtils::GetBuffer<TableBuffer>(argument, false))
			{
				for (const auto& [name, column] : table->GetColumns())
				{
					names.push_back(StringBucket::HashString(name));
					types.push_back(StringBucket::HashString(HorseIR::PrettyPrinter::PrettyString(column->GetType())));

					if (column == table->GetPrimaryKey())
					{
						attributes.push_back(StringBucket::HashString("primary"));
					}
					else
					{
						attributes.push_back(StringBucket::HashString(""));
					}
				}
			}
			else if (auto keyedTable = BufferUtils::GetBuffer<KeyedTableBuffer>(argument, false))
			{
				CUDA::Vector<std::uint64_t> kinds;
				
				auto keyTable = keyedTable->GetKey();
				for (const auto& [name, column] : keyTable->GetColumns())
				{
					kinds.push_back(StringBucket::HashString("key"));
					names.push_back(StringBucket::HashString(name));
					types.push_back(StringBucket::HashString(HorseIR::PrettyPrinter::PrettyString(column->GetType())));

					if (column == table->GetPrimaryKey())
					{
						attributes.push_back(StringBucket::HashString("primary"));
					}
					else
					{
						attributes.push_back(StringBucket::HashString(""));
					}
				}

				auto valueTable = keyedTable->GetValue();
				for (const auto& [name, column] : valueTable->GetColumns())
				{
					kinds.push_back(StringBucket::HashString("value"));
					names.push_back(StringBucket::HashString(name));
					types.push_back(StringBucket::HashString(HorseIR::PrettyPrinter::PrettyString(column->GetType())));

					if (column == table->GetPrimaryKey())
					{
						attributes.push_back(StringBucket::HashString("primary"));
					}
					else
					{
						attributes.push_back(StringBucket::HashString(""));
					}
				}

				columns.push_back({"kind", new TypedVectorBuffer(new TypedVectorData(new HorseIR::BasicType(HorseIR::BasicType::BasicKind::String), std::move(kinds)))});
			}
			else
			{
				Error("unsupported target type " + HorseIR::TypeUtils::TypeString(argument->GetType()));
			}

			columns.push_back({"name", new TypedVectorBuffer(new TypedVectorData(new HorseIR::BasicType(HorseIR::BasicType::BasicKind::String), std::move(names)))});
			columns.push_back({"type", new TypedVectorBuffer(new TypedVectorData(new HorseIR::BasicType(HorseIR::BasicType::BasicKind::String), std::move(types)))});
			columns.push_back({"attributes", new TypedVectorBuffer(new TypedVectorData(new HorseIR::BasicType(HorseIR::BasicType::BasicKind::String), std::move(attributes)))});

			return {new TableBuffer(columns)};
		}
		case HorseIR::BuiltinFunction::Primitive::Fetch:
		{
			return {BufferUtils::GetBuffer<EnumerationBuffer>(arguments.at(0))->GetValues()};
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

			const auto& likePatternString = StringBucket::RecoverString(patternData->GetValue(0));

			const auto size = stringData.size();
			CUDA::Vector<std::int8_t> likeData(size);

			auto likeKind = Utils::Options::GetAlgorithm_LikeKind();
			if (likeKind == Utils::Options::LikeKind::InternalLike)
			{
				for (auto i = 0u; i < size; ++i)
				{
					likeData[i] = Utils::String::Like(StringBucket::RecoverString(stringData[i]), likePatternString);
				}

			}
			else if (likeKind == Utils::Options::LikeKind::PCRELike)
			{
				// Transform from SQL like to regex
				//  - Escape: '.', '*', and '\'
				//  - Replace: '%' by '.*' (0 or more) and '_' by '.' (exactly 1)

				const auto likePatternSize = likePatternString.size();
				const char *likePattern = likePatternString.c_str();
				auto regexPattern = new char[likePatternSize * 2 + 2];

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

				jpcre2::select<char>::Regex regex(regexPatternString, PCRE2_DOTALL|PCRE2_ANCHORED|PCRE2_NO_UTF_CHECK, jpcre2::JIT_COMPILE);
				if (!regex)
				{
					Error("unable to compile regex pattern '" + regexPatternString + "' from like pattern '" + likePatternString + "'");
				}

				for (auto i = 0u; i < size; ++i)
				{
					likeData.at(i) = regex.match(&StringBucket::RecoverString(stringData.at(i)));
				}
			}

			return {new TypedVectorBuffer(new TypedVectorData<std::int8_t>(new HorseIR::BasicType(HorseIR::BasicType::BasicKind::Boolean), std::move(likeData)))};
		}
		case HorseIR::BuiltinFunction::Primitive::Print:
		{
			Utils::Logger::LogInfo(arguments.at(0)->DebugDump(), 0, true, Utils::Logger::NoPrefix);
			CUDA::Vector<std::int64_t> data({0});
			return {new TypedVectorBuffer(new TypedVectorData<std::int64_t>(new HorseIR::BasicType(HorseIR::BasicType::BasicKind::Int64), std::move(data)))};
		}
		case HorseIR::BuiltinFunction::Primitive::String:
		{
			auto string = arguments.at(0)->DebugDump();
			auto hash = StringBucket::HashString(string);
			CUDA::Vector<std::uint64_t> data({hash});
			return {new TypedVectorBuffer(new TypedVectorData<std::uint64_t>(new HorseIR::BasicType(HorseIR::BasicType::BasicKind::String), std::move(data)))};
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

		// GPU library
		case HorseIR::BuiltinFunction::Primitive::GPUOrderLib:
		{
			// GPU sort!

			GPU::SortEngine sortEngine(m_runtime, m_program);
			auto [indexBuffer, dataBuffer] = sortEngine.Sort({ std::begin(arguments), std::end(arguments) });

			// Data buffers can be deallocated as they are unused in a simple sort

			delete dataBuffer;
			return {indexBuffer};
		}
		case HorseIR::BuiltinFunction::Primitive::GPUGroupLib:
		{
			// GPU group!

			GPU::GroupEngine groupEngine(m_runtime, m_program);
			return {groupEngine.Group({ std::begin(arguments), std::end(arguments) })};
		}
		case HorseIR::BuiltinFunction::Primitive::GPUUniqueLib:
		{
			// GPU unique!

			GPU::UniqueEngine uniqueEngine(m_runtime, m_program);
			return {uniqueEngine.Unique({ std::begin(arguments), std::end(arguments) })};
		}
		case HorseIR::BuiltinFunction::Primitive::GPULoopJoinLib:
		{
			// GPU loop join!

			GPU::LoopJoinEngine joinEngine(m_runtime, m_program);
			return {joinEngine.Join({ std::begin(arguments), std::end(arguments) })};
		}
		case HorseIR::BuiltinFunction::Primitive::GPUHashJoinLib:
		{
			// GPU hash join!

			GPU::HashJoinEngine joinEngine(m_runtime, m_program);
			return {joinEngine.Join({ std::begin(arguments), std::end(arguments) })};
		}
		default:
		{
			Error("unimplemented");
		}
	}
}

}
