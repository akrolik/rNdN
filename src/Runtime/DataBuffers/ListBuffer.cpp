#include "Runtime/DataBuffers/ListBuffer.h"

#include "HorseIR/Utils/PrettyPrinter.h"

#include "Utils/Logger.h"

namespace Runtime {

void ListBuffer::AddCell(DataBuffer *cell)
{
	// Check that this is either the same type as the list contents or that the list is heterogenous

	m_type->AddElementType(cell->GetType()->Clone());
	m_cells.push_back(cell);
}

std::string ListBuffer::Description() const
{
	std::string description = HorseIR::PrettyPrinter::PrettyString(m_type) + "{";
	bool first = true;
	for (const auto& object : m_cells)
	{
		if (!first)
		{
			description += ", ";
		}
		first = false;
		description += object->Description();
	}
	return description + "}";
}

std::string ListBuffer::DebugDump() const
{
	std::string string = "[";
	for (const auto& cell : m_cells)
	{
		string += cell->DebugDump();
	}
	return (string + "]");
}

}
