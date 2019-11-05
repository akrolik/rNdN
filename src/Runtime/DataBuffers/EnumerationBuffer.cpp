#include "Runtime/DataBuffers/EnumerationBuffer.h"

#include "HorseIR/Utils/PrettyPrinter.h"

namespace Runtime {

EnumerationBuffer::EnumerationBuffer(VectorBuffer *keys, VectorBuffer *values) : ColumnBuffer(DataBuffer::Kind::Enumeration), m_keys(keys), m_values(values)
{
	// Form the type/shape

	m_shape = new Analysis::EnumerationShape(keys->GetShape(), values->GetShape());
	m_type = new HorseIR::EnumerationType(keys->GetType()->Clone());

	m_size = values->GetElementCount();
}

EnumerationBuffer::~EnumerationBuffer()
{
	delete m_type;
	delete m_shape;
}

std::string EnumerationBuffer::Description() const
{
	std::string description = HorseIR::PrettyPrinter::PrettyString(m_type) + "{";
	description += m_keys->Description() + " -> " + m_values->Description();
	return description + "}";
}

std::string EnumerationBuffer::DebugDump() const
{
	std::string string = "";
	for (auto i = 0ul; i < m_size; ++i)
	{
		string += DebugDump(i) + "\n";
	}
	return string;
}

std::string EnumerationBuffer::DebugDump(unsigned int index) const
{
	return m_keys->DebugDump(index) + " -> " + m_values->DebugDump(index);
}

}
