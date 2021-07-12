#include "Runtime/DataBuffers/KeyedTableBuffer.h"

#include "HorseIR/Utils/PrettyPrinter.h"
#include "HorseIR/Utils/TypeUtils.h"

#include "Utils/Chrono.h"
#include "Utils/Logger.h"

namespace Runtime {

KeyedTableBuffer::KeyedTableBuffer(TableBuffer *key, TableBuffer *value) : DataBuffer(DataBuffer::Kind::KeyedTable), m_key(key), m_value(value)
{
	// Form the type/shape

	auto keyShape = key->GetShape();
	auto valueShape = value->GetShape();
	m_shape = new HorseIR::Analysis::KeyedTableShape(keyShape, valueShape);

	m_type = new HorseIR::KeyedTableType();
}

KeyedTableBuffer::~KeyedTableBuffer()
{
	delete m_type;
	delete m_shape;
}

KeyedTableBuffer *KeyedTableBuffer::Clone() const
{
	return new KeyedTableBuffer(m_key->Clone(), m_value->Clone());
}

void KeyedTableBuffer::SetTag(const std::string& tag)
{
	DataBuffer::SetTag(tag);

	m_key->SetTag((tag == "") ? "" : tag + "_key");
	m_value->SetTag((tag == "") ? "" : tag + "_value");
}

void KeyedTableBuffer::ValidateCPU() const
{
	auto timeStart = Utils::Chrono::Start(TransferString("CPU ktable"));

	m_key->ValidateCPU();
	m_value->ValidateCPU();
	DataBuffer::ValidateCPU();

	Utils::Chrono::End(timeStart);
}

void KeyedTableBuffer::ValidateGPU() const
{
	auto timeStart = Utils::Chrono::Start(TransferString("GPU ktable"));

	m_key->ValidateGPU();
	m_value->ValidateGPU();
	DataBuffer::ValidateGPU();

	Utils::Chrono::End(timeStart);
}

std::string KeyedTableBuffer::Description() const
{
	std::string description = HorseIR::PrettyPrinter::PrettyString(m_type) + "{";
	description += m_key->Description() + " -> " + m_value->Description();
	return description + "}";
}

std::string KeyedTableBuffer::DebugDump(unsigned int indent, bool preindent) const
{
	std::string indentString(indent * Utils::Logger::IndentSize, ' ');

	std::string string;
	if (!preindent)
	{
		string += indentString;
	}

	string += "{\n";
	string += m_key->DebugDump(indent + 1) + "\n";
	string += indentString + "->\n";
	string += m_value->DebugDump(indent + 1) + "\n";
	string += indentString + "}"; 
	return string;
}

void KeyedTableBuffer::Clear(ClearMode mode)
{
	if (mode == ClearMode::Zero)
	{
		m_key->Clear(mode);
		m_value->Clear(mode);
	}
}

}
