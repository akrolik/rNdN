#pragma once

#include "PTX/Tree/Directives/Directive.h"

namespace PTX {

class FileDirective : public Directive
{
public:
	FileDirective(unsigned int index, const std::string& name) : FileDirective(index, name, 0, 0) {}
	FileDirective(unsigned int index, const std::string& name, unsigned int timestamp, unsigned int filesize) : m_index(index), m_name(name), m_timestamp(timestamp), m_filesize(filesize) {}

	// Properties

	unsigned int GetIndex() const { return m_index; }
	void SetIndex(unsigned int index) { m_index = index; }

	const std::string& GetName() const { return m_name; }
	void SetName(const std::string& name) { m_name = name; }

	unsigned int GetTimestamp() const { return m_timestamp; }
	void SetTimestamp(unsigned int timestamp) { m_timestamp = timestamp; }

	unsigned int GetFilesize() const { return m_filesize; }
	void SetFilesize(unsigned int filesize) { m_filesize = filesize; }

	// Formatting

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

	// Visitors

	void Accept(Visitor &visitor) override { visitor.Visit(this); }
	void Accept(ConstVisitor &visitor) const override { visitor.Visit(this); }

	void Accept(HierarchicalVisitor &visitor) override
	{
		visitor.VisitIn(this);
		visitor.VisitOut(this);
	}

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
