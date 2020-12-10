#pragma once

#include <vector>

#include "PTX/Tree/Tree.h"

namespace PTX {
namespace Analysis {

class BasicBlock
{
public:
	BasicBlock(Label *label) : m_label(label) {}

	Label *GetLabel() const { return m_label; }
	void SetLabel(Label *label) { m_label = label; }

	const std::vector<Statement *> GetStatements() const { return m_statements; }
	void AddStatement(Statement *statement) { m_statements.push_back(statement); }

	std::size_t GetSize() const { return m_statements.size(); }

	std::string ToDOTString() const
	{
		if (m_statements.size() == 0)
		{
			return "%empty%";
		}

		if (m_statements.size() <= 3)
		{
			std::string string;
			for (const auto& statement : m_statements)
			{
				// string += statement->ToString(0) + "\\l";
			}
			return string;
		}

		std::string string;
		// string += m_statements.at(0)->ToString(0) + "\\l";
		// string += m_statements.at(1)->ToString(0) + "\\l";
		string += "[...]\\l";
		// string += m_statements.back()->ToString(0) + "\\l";
		return string;
	}

private:
	Label *m_label = nullptr;
	std::vector<Statement *> m_statements;
};

}
}
