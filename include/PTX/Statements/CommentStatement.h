#pragma once

#include "PTX/Statements/Statement.h"

namespace PTX {

class CommentStatement : public Statement
{
public:
	CommentStatement(const std::string& comment, bool multiline = false) : m_comment(comment), m_multiline(multiline) {}

	std::string ToString(unsigned int indentation) const override
	{
		std::string indentString = std::string(indentation, '\t');
		if (m_multiline)
		{
			return indentString + "/*\n" + indentString + " * " + ReplaceString(m_comment, "\n", "\n" + indentString + " * ") + "\n" + indentString + " */";
		}
		return indentString + "// " + ReplaceString(m_comment, "\n", "\n" + indentString + "// ");
	}

	json ToJSON() const override
	{
		json j;
		j["kind"] = "PTX::CommentStatement";
		j["comment"] = m_comment;
		j["mulitline"] = m_multiline;
		return j;
	}

	// Visitors

	void Accept(ConstHierarchicalVisitor& visitor) const override
	{
		visitor.VisitIn(this);
		visitor.VisitOut(this);
	}

private:
	static std::string ReplaceString(std::string subject, const std::string& search, const std::string& replace)
	{
		size_t pos = 0;
		while ((pos = subject.find(search, pos)) != std::string::npos)
		{
			subject.replace(pos, search.length(), replace);
			pos += replace.length();
		}
		return subject;
	}

	std::string m_comment;
	bool m_multiline = false;
};

}
