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

	Program *Clone() const override
	{
		std::vector<Module *> modules;
		for (const auto& module : m_modules)
		{
			modules.push_back(module->Clone());
		}
		return new Program(modules);
	}

	// Contents

	std::vector<const Module *> GetModules() const
	{
		return { std::begin(m_modules), std::end(m_modules) };
	}
	std::vector<Module *>& GetModules() { return m_modules; }

	void AddModule(Module *module) { m_modules.push_back(module); }
	void SetModules(const std::vector<Module *> modules) { m_modules = modules; }

	// Symbol table

	const SymbolTable *GetSymbolTable() const { return m_symbolTable; }
	SymbolTable *GetSymbolTable() { return m_symbolTable; }
	void SetSymbolTable(SymbolTable *symbolTable) { m_symbolTable = symbolTable; }

	// Visitors

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
