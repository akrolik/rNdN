#pragma once

#include <string>
#include <vector>

#include "PTX/Tree/Tree.h"

namespace PTX {
namespace Analysis {

class BasicBlock
{
public:
	BasicBlock(const std::string& name) : m_name(name) {}

	const std::string& GetName() const { return m_name; }
	void SetName(const std::string& name) { m_name = name; }

	const std::vector<const PTX::Statement *> GetStatements() const { return m_statements; }
	void AddStatement(const PTX::Statement *statement) { m_statements.push_back(statement); }

private:
	std::string m_name;
	std::vector<const PTX::Statement *> m_statements;
};

}
}
