#pragma once

#include "PTX/Tree/Directives/Directive.h"
#include "PTX/Tree/Statements/DirectiveStatement.h"

namespace PTX {

class LocationDirective : public Directive, public DirectiveStatement
{
public:
	LocationDirective(const FileDirective *file, unsigned int line, unsigned int column = 1) : m_file(file), m_line(line), m_column(column) {}

	const FileDirective *GetFile() const { return m_file; }
	unsigned int GetLine() const { return m_line; }
	unsigned int GetColum() const { return m_column; }

	std::string ToString(unsigned int indentation) const override
	{
		std::string string = std::string(indentation, '\t');
		return string + ".loc " + std::to_string(m_file->GetIndex()) + " " + std::to_string(m_line) + " " + std::to_string(m_column);
	}

	json ToJSON() const override
	{
		json j;
		j["kind"] = "PTX::LocationDirective";
		j["file"] = std::to_string(m_file->GetIndex());
		j["line"] = m_line;
		j["column"] = m_column;
		return j;
	}

	// Visitor

	void Accept(ConstHierarchicalVisitor &visitor) const override
	{
		visitor.VisitIn(this);
		visitor.VisitOut(this);
	}

private:
	const FileDirective *m_file = nullptr;

	unsigned int m_line = 0;
	unsigned int m_column = 1;
};

}
