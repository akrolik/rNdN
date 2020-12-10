#pragma once

#include "PTX/Tree/Directives/Directive.h"
#include "PTX/Tree/Statements/DirectiveStatement.h"

namespace PTX {

class LocationDirective : public Directive, public DirectiveStatement
{
public:
	LocationDirective(FileDirective *file, unsigned int line, unsigned int column = 1) : m_file(file), m_line(line), m_column(column) {}

	// Properties

	const FileDirective *GetFile() const { return m_file; }
	FileDirective *GetFile() { return m_file; }
	void SetFile(FileDirective *file) { m_file = file; }

	unsigned int GetLine() const { return m_line; }
	void SetLine(unsigned int line) { m_line = line; }

	unsigned int GetColumn() const { return m_column; }
	void SetColumn(unsigned int column) { m_column = column; }

	// Formatting

	json ToJSON() const override
	{
		json j;
		j["kind"] = "PTX::LocationDirective";
		j["file"] = std::to_string(m_file->GetIndex());
		j["line"] = m_line;
		j["column"] = m_column;
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
	FileDirective *m_file = nullptr;

	unsigned int m_line = 0;
	unsigned int m_column = 1;
};

}
