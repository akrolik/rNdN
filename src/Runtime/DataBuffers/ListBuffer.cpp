#include "Runtime/DataBuffers/ListBuffer.h"

#include "HorseIR/Utils/PrettyPrinter.h"

#include "Utils/Logger.h"

namespace Runtime {

ListBuffer *ListBuffer::Create(const HorseIR::ListType *type, const Analysis::ListShape *shape)
{
	auto elementTypes = type->GetElementTypes();
	auto elementShapes = shape->GetElementShapes();

	auto typeCount = elementTypes.size();
	auto shapeCount = elementShapes.size();

	if (typeCount != 1 && typeCount != shapeCount)
	{
		Utils::Logger::LogError("Mismatch between list type and shape cell count [" + std::to_string(typeCount) + " != " + std::to_string(shapeCount) + "]");
	}

	std::vector<DataBuffer *> cellBuffers;
	for (auto i = 0u; i < shapeCount; ++i)
	{
		auto elementType = (typeCount == 1) ? elementTypes.at(0) : elementTypes.at(i);
		cellBuffers.push_back(DataBuffer::Create(elementType, elementShapes.at(i)));
	}

	return new ListBuffer(cellBuffers);
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
