#pragma once

#include <vector>

#include "PTX/Tree/Tree.h"

namespace PTX {
namespace Analysis {

class BasicBlock
{
public:
	BasicBlock(Label *label) : m_label(label) {}

	// Properties

	Label *GetLabel() const { return m_label; }
	void SetLabel(Label *label) { m_label = label; }

	const std::vector<Statement *> GetStatements() const { return m_statements; }
	void AddStatement(Statement *statement) { m_statements.push_back(statement); }

	std::size_t GetSize() const { return m_statements.size(); }

	// Formatting

	std::string ToDOTString() const;

private:
	Label *m_label = nullptr;
	std::vector<Statement *> m_statements;
};

}
}
