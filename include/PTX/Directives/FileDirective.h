#pragma once

#include "PTX/Directives/Directive.h"

namespace PTX {

class FileDirective : public Directive
{
public:
	FileDirective(unsigned int index, const std::string& name) : FileDirective(index, name, 0, 0) {}
	FileDirective(unsigned int index, const std::string& name, unsigned int timestamp, unsigned int filesize) : m_index(index), m_name(name), m_timestamp(timestamp), m_filesize(filesize) {}

	unsigned int GetIndex() const { return m_index; }
	const std::string& GetName() const { return m_name; }
	unsigned int GetTimestamp() const { return m_timestamp; }
	unsigned int GetFilesize() const { return m_filesize; }

	std::string ToString(unsigned int indentation) const override
	{
		std::string string = std::string(indentation, '\t');
		string += ".file " + std::to_string(m_index) + " \"" + m_name + "\"";

		if (m_timestamp > 0 || m_filesize > 0)
		{
			string += ", " + std::to_string(m_timestamp) + ", " + std::to_string(m_filesize);
		}

		return string;
	}

	json ToJSON() const override
	{
		json j;
		j["kind"] = "PTX::FileDirective";
		j["index"] = m_index;
		j["name"] = m_name;
		if (m_timestamp > 0 || m_filesize > 0)
		{
			j["timestamp"] = m_timestamp;
			j["filesize"] = m_filesize;
		}
		return j;
	}

	// Visitor

	void Accept(ConstHierarchicalVisitor &visitor) const override
	{
		visitor.VisitIn(this);
		visitor.VisitOut(this);
	}

private:
	unsigned int m_index = 0;
	std::string m_name;
	unsigned int m_timestamp = 0;
	unsigned int m_filesize = 0;
};

}
