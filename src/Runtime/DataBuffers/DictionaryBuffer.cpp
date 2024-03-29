#include "Runtime/DataBuffers/DictionaryBuffer.h"

#include "HorseIR/Analysis/Shape/ShapeUtils.h"

#include "HorseIR/Utils/PrettyPrinter.h"
#include "HorseIR/Utils/TypeUtils.h"

#include "Utils/Chrono.h"
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
	auto valuesShape = values->GetShape();
	m_shape = new HorseIR::Analysis::DictionaryShape(keysShape, valuesShape);

	auto keysType = keys->GetType()->Clone();
	auto valuesType = HorseIR::TypeUtils::GetReducedType(values->GetType()->GetElementTypes())->Clone();
	m_type = new HorseIR::DictionaryType(keysType, valuesType);
}

DictionaryBuffer::~DictionaryBuffer()
{
	delete m_type;
	delete m_shape;
}

DictionaryBuffer *DictionaryBuffer::Clone() const
{
	return new DictionaryBuffer(m_keys->Clone(), m_values->Clone());
}

void DictionaryBuffer::SetTag(const std::string& tag)
{
	DataBuffer::SetTag(tag);

	m_keys->SetTag((tag == "") ? "" : tag + "_keys");
	m_values->SetTag((tag == "") ? "" : tag + "_values");
}

void DictionaryBuffer::RequireCPUConsistent(bool exclusive) const
{
	auto timeStart = Utils::Chrono::Start(TransferString("CPU dictionary"));

	m_keys->RequireCPUConsistent(exclusive);
	m_values->RequireCPUConsistent(exclusive);

	DataBuffer::SetCPUConsistent(exclusive);

	Utils::Chrono::End(timeStart);
}

void DictionaryBuffer::RequireGPUConsistent(bool exclusive) const
{
	auto timeStart = Utils::Chrono::Start(TransferString("GPU dictionary"));

	m_keys->RequireGPUConsistent(exclusive);
	m_values->RequireGPUConsistent(exclusive);

	DataBuffer::SetGPUConsistent(exclusive);

	Utils::Chrono::End(timeStart);
}

std::string DictionaryBuffer::Description() const
{
	std::string description = HorseIR::PrettyPrinter::PrettyString(m_type) + "{";
	description += m_keys->Description() + " -> " + m_values->Description();
	return description + "}";
}

std::string DictionaryBuffer::DebugDump(unsigned int indent, bool preindent) const
{
	std::string indentString(indent * Utils::Logger::IndentSize, ' ');
	std::string indentStringP1((indent + 1) * Utils::Logger::IndentSize, ' ');

	std::string string;
	if (!preindent)
	{
		string += indentString;
	}

	string += "{";
	if (m_size > 0)
	{
		string += "\n";
		for (auto i = 0ul; i < m_size; ++i)
		{
			string += indentStringP1 + "[" + m_keys->DebugDumpElement(i) + "] -> ";
			string += m_values->GetCell(i)->DebugDump(indent + 1, true) + "\n";
		}
		string += indentString;
	}
	return string + "}";
}

void DictionaryBuffer::Clear(ClearMode mode)
{
	if (mode == ClearMode::Zero)
	{
		m_keys->Clear(mode);
		m_values->Clear(mode);
	}
}

}
