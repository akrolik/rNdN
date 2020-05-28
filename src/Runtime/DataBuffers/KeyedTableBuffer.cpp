#include "Runtime/DataBuffers/KeyedTableBuffer.h"

#include "Analysis/Shape/ShapeUtils.h"

#include "HorseIR/Utils/PrettyPrinter.h"
#include "HorseIR/Utils/TypeUtils.h"

#include "Utils/Logger.h"

namespace Runtime {

KeyedTableBuffer::KeyedTableBuffer(TableBuffer *key, TableBuffer *value) : DataBuffer(DataBuffer::Kind::KeyedTable), m_key(key), m_value(value)
{
	// Form the type/shape

	auto keyShape = key->GetShape();
	auto valueShape = value->GetShape();
	m_shape = new Analysis::KeyedTableShape(keyShape, valueShape);

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

void KeyedTableBuffer::ValidateCPU(bool recursive) const
{
	DataBuffer::ValidateCPU(recursive);
	if (recursive)
	{
		m_key->ValidateCPU(true);
		m_value->ValidateCPU(true);
	}
}

void KeyedTableBuffer::ValidateGPU(bool recursive) const
{
	DataBuffer::ValidateGPU(recursive);
	if (recursive)
	{
		m_key->ValidateGPU(true);
		m_value->ValidateGPU(true);
	}
}

std::string KeyedTableBuffer::Description() const
{
	std::string description = HorseIR::PrettyPrinter::PrettyString(m_type) + "{";
	description += m_key->Description() + " -> " + m_value->Description();
	return description + "}";
}

std::string KeyedTableBuffer::DebugDump() const
{
	std::string string = "{";
	string += m_key->DebugDump() + "\n -> \n";
	string += m_value->DebugDump();
	string += "\n}"; 
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
