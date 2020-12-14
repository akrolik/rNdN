#pragma once

#include <sstream>

namespace Runtime { class DataBuffer; }

namespace HorseIR {
namespace Analysis {

class DataObject
{
public:
	DataObject() : m_objectID(m_index++) {}
	DataObject(unsigned int objectID, const Runtime::DataBuffer *buffer) : m_objectID(objectID), m_buffer(buffer) {}

	unsigned int GetObjectID() const { return m_objectID; }

	void SetDataBuffer(const Runtime::DataBuffer *buffer) { m_buffer = buffer; }
	const Runtime::DataBuffer *GetDataBuffer() const { return m_buffer; }

	bool operator==(const DataObject& other) const
	{
		return (m_objectID == other.m_objectID);
	}

	bool operator!=(const DataObject& other) const
	{
		return !(*this == other);
	}

	std::string ToString() const { return "ID_" + std::to_string(m_objectID); }

	friend std::ostream& operator<<(std::ostream& os, const DataObject& object);

private:
	unsigned int m_objectID = 0;
	const Runtime::DataBuffer *m_buffer = nullptr;

	static unsigned int m_index;
};

inline std::ostream& operator<<(std::ostream& os, const DataObject& object)
{
	os << object.ToString();
	return os;
}

}
}
