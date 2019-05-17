#pragma once

#include <vector>

#include "HorseIR/Tree/Node.h"

#include "HorseIR/Tree/Module.h"

#include "HorseIR/Traversal/ConstVisitor.h"
#include "HorseIR/Traversal/ConstHierarchicalVisitor.h"
#include "HorseIR/Traversal/Visitor.h"
#include "HorseIR/Traversal/HierarchicalVisitor.h"

class SymbolTable;

namespace HorseIR {

class Program : public Node
{
public:
	Program(const std::vector<Module *>& modules) : m_modules(modules) {}

	const std::vector<Module *>& GetModules() const { return m_modules; }
	void AddModule(Module *module) { m_modules.push_back(module); }
	void SetModules(const std::vector<Module *> modules) { m_modules = modules; }

	//TODO: Remove this
	SymbolTable *GetSymbolTable() const { return m_symbolTable; }
	void SetSymbolTable(SymbolTable *symbolTable) { m_symbolTable = symbolTable; }

	void Accept(Visitor &visitor) override { visitor.Visit(this); }
	void Accept(ConstVisitor &visitor) const override { visitor.Visit(this); }

	void Accept(HierarchicalVisitor &visitor) override
	{
		if (visitor.VisitIn(this))
		{
			for (auto& module : m_modules)
			{
				module->Accept(visitor);
			}
		}
		visitor.VisitOut(this);
	}

	void Accept(ConstHierarchicalVisitor &visitor) const override
	{
		if (visitor.VisitIn(this))
		{
			for (const auto& module : m_modules)
			{
				module->Accept(visitor);
			}
		}
		visitor.VisitOut(this);
	}

protected:
	std::vector<Module *> m_modules;

	SymbolTable *m_symbolTable = nullptr;
};

}
