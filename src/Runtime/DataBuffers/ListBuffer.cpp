#include "Runtime/DataBuffers/ListBuffer.h"

#include "Analysis/Shape/ShapeUtils.h"

#include "HorseIR/Utils/PrettyPrinter.h"

#include "Utils/Logger.h"

namespace Runtime {

ListBuffer *ListBuffer::CreateEmpty(const HorseIR::ListType *type, const Analysis::ListShape *shape)
{
	auto elementTypes = type->GetElementTypes();
	auto elementShapes = shape->GetElementShapes();

	auto typeCount = elementTypes.size();
	auto shapeCount = elementShapes.size();

	if (const auto listSize = Analysis::ShapeUtils::GetSize<Analysis::Shape::ConstantSize>(shape->GetListSize()))
	{
		shapeCount = listSize->GetValue();
	}

	if (typeCount != 1 && typeCount != shapeCount)
	{
		Utils::Logger::LogError("Mismatch between list type and shape cell count [" + HorseIR::PrettyPrinter::PrettyString(type) + "; " + Analysis::ShapeUtils::ShapeString(shape) + "]");
	}

	std::vector<DataBuffer *> cellBuffers;
	for (auto i = 0u; i < shapeCount; ++i)
	{
		auto elementType = (typeCount == 1) ? elementTypes.at(0) : elementTypes.at(i);
		auto elementShape = (elementShapes.size() == 1) ? elementShapes.at(0) : elementShapes.at(i);

		cellBuffers.push_back(DataBuffer::CreateEmpty(elementType, elementShape));
	}

	return new ListBuffer(cellBuffers);
}

ListBuffer::ListBuffer(const std::vector<DataBuffer *>& cells) : DataBuffer(DataBuffer::Kind::List), m_cells(cells)
{
	std::vector<HorseIR::Type *> cellTypes;
	std::vector<const Analysis::Shape *> cellShapes;
	for (const auto& cell : cells)
	{
		cellTypes.push_back(cell->GetType()->Clone());
		cellShapes.push_back(cell->GetShape());
	}
	m_type = new HorseIR::ListType(cellTypes);
	m_shape = new Analysis::ListShape(new Analysis::Shape::ConstantSize(cells.size()), cellShapes);

	m_cpuConsistent = true; // Always CPU consistent
}

ListBuffer::~ListBuffer()
{
	delete m_type;
	delete m_shape;
	
	delete m_gpuBuffer;
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
