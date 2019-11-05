#include "Runtime/DataBuffers/DictionaryBuffer.h"

#include "Analysis/Shape/ShapeUtils.h"

#include "HorseIR/Utils/PrettyPrinter.h"
#include "HorseIR/Utils/TypeUtils.h"

#include "Utils/Logger.h"

namespace Runtime {

DictionaryBuffer::DictionaryBuffer(VectorBuffer *keys, ListBuffer *values) : DataBuffer(DataBuffer::Kind::Dictionary), m_keys(keys), m_values(values)
{
	// Check key count == value count

	auto keysCount = keys->GetElementCount();
	auto valuesCount = values->GetCellCount();
	if (keysCount != valuesCount)
	{
		Utils::Logger::LogError("Dictionary has different number of keys and values [" + std::to_string(keysCount) + " != " + std::to_string(valuesCount) + "]");
	}
	m_size = keysCount;

	// Form the type/shape

	auto keysShape = keys->GetShape();
	auto valuesShape = Analysis::ShapeUtils::MergeShapes(values->GetShape()->GetElementShapes());
	m_shape = new Analysis::DictionaryShape(keysShape, valuesShape);

	auto keysType = keys->GetType()->Clone();
	auto valuesType = HorseIR::TypeUtils::GetReducedType(values->GetType()->GetElementTypes());
	m_type = new HorseIR::DictionaryType(keysType, valuesType);
}

DictionaryBuffer::~DictionaryBuffer()
{
	delete m_type;
	delete m_shape;
}

std::string DictionaryBuffer::Description() const
{
	std::string description = HorseIR::PrettyPrinter::PrettyString(m_type) + "{";
	description += m_keys->Description() + " -> " + m_values->Description();
	return description + "}";
}

std::string DictionaryBuffer::DebugDump() const
{
	std::string string = "";
	for (auto i = 0ul; i < m_size; ++i)
	{
		string += m_keys->DebugDump(i) + " -> ";
		string += m_values->GetCell(i)->DebugDump();
		string += "\n";
	}
	return string;
}

}
