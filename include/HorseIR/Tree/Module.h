#pragma once

#include <string>
#include <vector>

#include "HorseIR/Tree/Node.h"

#include "HorseIR/Semantics/SymbolTable/SymbolTable.h"
#include "HorseIR/Tree/ModuleContent.h"

#include "HorseIR/Traversal/ConstVisitor.h"
#include "HorseIR/Traversal/ConstHierarchicalVisitor.h"
#include "HorseIR/Traversal/Visitor.h"
#include "HorseIR/Traversal/HierarchicalVisitor.h"

namespace HorseIR {

class Module : public Node
{
public:
	Module(const std::string& name, const std::vector<ModuleContent *>& contents) : m_name(name), m_contents(contents) {}

	Module *Clone() const override
	{
		std::vector<ModuleContent *> contents;
		for (const auto& content : m_contents)
		{
			contents.push_back(content->Clone());
		}
		return new Module(m_name, contents);
	}

	const std::string& GetName() const { return m_name; }
	void SetName(const std::string& name) { m_name = name; }

	const std::vector<ModuleContent *>& GetContents() const { return m_contents; }
	void AddContent(ModuleContent *content) { m_contents.push_back(content); }
	void SetContents(const std::vector<ModuleContent *>& contents) { m_contents = contents; }

	SymbolTable *GetSymbolTable() const { return m_symbolTable; }
	void SetSymbolTable(SymbolTable *symbolTable) { m_symbolTable = symbolTable; }

	void Accept(Visitor &visitor) override { visitor.Visit(this); }
	void Accept(ConstVisitor &visitor) const override { visitor.Visit(this); }

	void Accept(HierarchicalVisitor &visitor) override
	{
		if (visitor.VisitIn(this))
		{
			for (auto& content : m_contents)
			{
				content->Accept(visitor);
			}
		}
		visitor.VisitOut(this);
	}

	void Accept(ConstHierarchicalVisitor &visitor) const override
	{
		if (visitor.VisitIn(this))
		{
			for (const auto& content : m_contents)
			{
				content->Accept(visitor);
			}
		}
		visitor.VisitOut(this);
	}

protected:
	std::string m_name;
	std::vector<ModuleContent *> m_contents;
	
	SymbolTable *m_symbolTable = nullptr;
};

}
