#pragma once

#include <vector>

#include "PTX/Tree/Tree.h"

namespace PTX {
namespace Analysis {

class BasicBlock
{
public:
	BasicBlock(Label *label) : m_label(label) {}

	// Labels

	const Label *GetLabel() const { return m_label; }
	Label *GetLabel() { return m_label; }

	void SetLabel(Label *label) { m_label = label; }

	// Statements

	std::vector<const Statement *> GetStatements() const
	{
		return { std::begin(m_statements), std::end(m_statements) };
	}
	std::vector<Statement *>& GetStatements() { return m_statements; }

	void AddStatement(Statement *statement) { m_statements.push_back(statement); }
	void SetStatements(const std::vector<Statement *>& statements) { m_statements = statements; }

	std::size_t GetSize() const { return m_statements.size(); }

	// Formatting

	std::string ToDOTString() const;

private:
	Label *m_label = nullptr;
	std::vector<Statement *> m_statements;
};

}
}
