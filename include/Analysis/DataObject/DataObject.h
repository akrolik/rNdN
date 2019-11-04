#pragma once

#include <sstream>

namespace Analysis {

class DataObject
{
public:
	DataObject() : m_objectID(m_index++) {}

	unsigned int GetObjectID() const { return m_objectID; }

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

	static unsigned int m_index;
};

inline std::ostream& operator<<(std::ostream& os, const DataObject& object)
{
	os << object.ToString();
	return os;
}

}
