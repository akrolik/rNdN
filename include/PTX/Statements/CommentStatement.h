#pragma once

#include "PTX/Statements/Statement.h"

namespace PTX {

class CommentStatement : public Statement
{
public:
	CommentStatement(std::string comment, bool multiline = false) : m_comment(comment), m_multiline(multiline) {}

	std::string ToString() const override
	{
		if (m_multiline)
		{
			return "/*\n" + m_comment + "\n*/";
		}
		return "// " + m_comment;
	}

	std::string Terminator() const override
	{
		return "";
	}

	json ToJSON() const override
	{
		json j;
		j["kind"] = "PTX::CommentStatement";
		j["comment"] = m_comment;
		j["mulitline"] = m_multiline;
		return j;
	}

private:
	std::string m_comment;
	bool m_multiline = false;
};

}
