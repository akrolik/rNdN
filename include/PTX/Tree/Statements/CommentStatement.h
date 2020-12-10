#pragma once

#include "PTX/Tree/Statements/Statement.h"

namespace PTX {

class CommentStatement : public Statement
{
public:
	CommentStatement(const std::string& comment, bool multiline = false) : m_comment(comment), m_multiline(multiline) {}

	// Properties

	const std::string& GetComment() const { return m_comment; }
	void SetComment(const std::string& comment) { m_comment = comment; }

	bool IsMultiline() const { return m_multiline; }
	void SetMultiline(bool multiline) { m_multiline = multiline; }

	// Formatting

	json ToJSON() const override
	{
		json j;
		j["kind"] = "PTX::CommentStatement";
		j["comment"] = m_comment;
		j["mulitline"] = m_multiline;
		return j;
	}

	// Visitors

	void Accept(Visitor& visitor) override { visitor.Visit(this); }
	void Accept(ConstVisitor& visitor) const override { visitor.Visit(this); }

	void Accept(HierarchicalVisitor& visitor) override
	{
		visitor.VisitIn(this);
		visitor.VisitOut(this);
	}

	void Accept(ConstHierarchicalVisitor& visitor) const override
	{
		visitor.VisitIn(this);
		visitor.VisitOut(this);
	}

private:
	std::string m_comment;
	bool m_multiline = false;
};

}
