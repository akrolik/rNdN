#include "Runtime/DataBuffers/EnumerationBuffer.h"

#include "HorseIR/Utils/PrettyPrinter.h"

namespace Runtime {

EnumerationBuffer::EnumerationBuffer(VectorBuffer *keys, VectorBuffer *values, TypedVectorBuffer<std::int64_t> *indexes) : ColumnBuffer(DataBuffer::Kind::Enumeration), m_keys(keys), m_values(values), m_indexes(indexes)
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

EnumerationBuffer *EnumerationBuffer::Clone() const
{
	return new EnumerationBuffer(m_keys->Clone(), m_values->Clone(), m_indexes->Clone());
}

void EnumerationBuffer::ValidateCPU(bool recursive) const
{
	ColumnBuffer::ValidateCPU(recursive);
	if (recursive)
	{
		m_keys->ValidateCPU(true);
		m_values->ValidateCPU(true);
		m_indexes->ValidateCPU(true);
	}
}

void EnumerationBuffer::ValidateGPU(bool recursive) const
{
	ColumnBuffer::ValidateGPU(recursive);
	if (recursive)
	{
		m_keys->ValidateGPU(true);
		m_values->ValidateGPU(true);
		m_indexes->ValidateGPU(true);
	}
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
	return m_values->DebugDump(index) + " -> [" + m_indexes->DebugDump(index) + "]";
}

void EnumerationBuffer::Clear(ClearMode mode)
{
	if (mode == ClearMode::Zero)
	{
		m_keys->Clear(mode);
		m_values->Clear(mode);
	}
}

}
