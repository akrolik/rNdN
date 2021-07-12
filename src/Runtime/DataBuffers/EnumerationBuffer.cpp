#include "Runtime/DataBuffers/EnumerationBuffer.h"

#include "HorseIR/Utils/PrettyPrinter.h"

#include "Utils/Chrono.h"

namespace Runtime {

EnumerationBuffer::EnumerationBuffer(VectorBuffer *keys, VectorBuffer *values, TypedVectorBuffer<std::int64_t> *indexes) : ColumnBuffer(DataBuffer::Kind::Enumeration), m_keys(keys), m_values(values), m_indexes(indexes)
{
	// Form the type/shape

	m_shape = new HorseIR::Analysis::EnumerationShape(keys->GetShape(), values->GetShape());
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

void EnumerationBuffer::SetTag(const std::string& tag)
{
	ColumnBuffer::SetTag(tag);

	m_values->SetTag((tag == "") ? "" : tag + "_values");
	m_indexes->SetTag((tag == "") ? "" : tag + "_indexes");
}

void EnumerationBuffer::ValidateCPU() const
{
	auto timeStart = Utils::Chrono::Start(TransferString("CPU enum"));

	m_keys->ValidateCPU();
	m_values->ValidateCPU();
	m_indexes->ValidateCPU();
	ColumnBuffer::ValidateCPU();

	Utils::Chrono::End(timeStart);
}

void EnumerationBuffer::ValidateGPU() const
{
	auto timeStart = Utils::Chrono::Start(TransferString("GPU enum"));

	m_keys->ValidateGPU();
	m_values->ValidateGPU();
	m_indexes->ValidateGPU();
	ColumnBuffer::ValidateGPU();

	Utils::Chrono::End(timeStart);
}

std::string EnumerationBuffer::Description() const
{
	std::string description = HorseIR::PrettyPrinter::PrettyString(m_type) + "{";
	description += m_keys->Description() + " -> " + m_values->Description();
	return description + "}";
}

std::string EnumerationBuffer::DebugDump(unsigned int indent, bool preindent) const
{
	std::string indentString(indent * Utils::Logger::IndentSize, ' ');
	std::string indentStringP1((indent + 1) * Utils::Logger::IndentSize, ' ');

	std::string string;
	if (!preindent)
	{
		string += indentString;
	}

	string += "[";
	if (m_size > 0)
	{
		string += "\n";
		for (auto i = 0ul; i < m_size; ++i)
		{
			string += indentStringP1 + DebugDumpElement(i, indent + 1, true) + "\n";
		}
		string += indentString;
	}
	return string + "]";
}

std::string EnumerationBuffer::DebugDumpElement(unsigned int index, unsigned int indent, bool preindent) const
{
	std::string string;
	if (!preindent)
	{
		string += std::string(indent * Utils::Logger::IndentSize, ' ');
	}
	return string + m_values->DebugDump(index) + " -> [" + m_indexes->DebugDump(index) + "]";
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
