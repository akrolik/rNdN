#pragma once

#include <string>
#include <vector>

#include "HorseIR/Tree/Node.h"

#include "HorseIR/Traversal/Visitor.h"
#include "HorseIR/Tree/Module.h"

class SymbolTable;

namespace HorseIR {

class Program : public Node
{
public:
	Program(const std::vector<Module *>& modules) : m_modules(modules) {}

	const std::vector<Module *>& GetModules() const { return m_modules; }
	void AddModule(Module *module) { m_modules.push_back(module); }

	SymbolTable *GetSymbolTable() const { return m_symbolTable; }
	void SetSymbolTable(SymbolTable *symbolTable) { m_symbolTable = symbolTable; }

	std::string ToString() const override
	{
		std::string code;
		for (const auto& module : m_modules)
		{
			code += module->ToString() + "\n";
		}
		return code;
	}

	void Accept(Visitor &visitor) override { visitor.Visit(this); }

private:
	std::vector<Module *> m_modules;

	SymbolTable *m_symbolTable = nullptr;
};

}
