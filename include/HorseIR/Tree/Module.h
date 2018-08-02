#pragma once

#include <string>
#include <vector>

#include "HorseIR/Tree/Node.h"

#include "HorseIR/Traversal/Visitor.h"
#include "HorseIR/Tree/ModuleContent.h"

class SymbolTable;

namespace HorseIR {

class Module : public Node
{
public:
	Module(const std::string& name, const std::vector<ModuleContent *>& contents) : m_name(name), m_contents(contents) {}

	const std::string& GetName() const { return m_name; }
	void SetName(const std::string& name) { m_name = name; }

	const std::vector<ModuleContent *>& GetContents() { return m_contents; }
	void AddContent(ModuleContent *content) { m_contents.push_back(content); }

	SymbolTable *GetSymbolTable() const { return m_symbolTable; }
	void SetSymbolTable(SymbolTable *symbolTable) { m_symbolTable = symbolTable; }

	std::string ToString() const override
	{
		std::string code = "module " + m_name + " {\n";

		for (const auto& contents : m_contents)
		{
			code += "\t" + contents->ToString() + "\n";
		}

		return code + "}";
	}

	void Accept(Visitor &visitor) override { visitor.Visit(this); }

private:
	std::string m_name;
	std::vector<ModuleContent *> m_contents;
	
	SymbolTable *m_symbolTable = nullptr;
};

}
