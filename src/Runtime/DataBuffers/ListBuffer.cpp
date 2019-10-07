#include "Runtime/DataBuffers/ListBuffer.h"

#include "HorseIR/Utils/PrettyPrinter.h"

#include "Utils/Logger.h"

namespace Runtime {

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
