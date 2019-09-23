#pragma once

#include <vector>

#include "HorseIR/Tree/Statements/Statement.h"

#include "HorseIR/Traversal/ConstVisitor.h"
#include "HorseIR/Traversal/ConstHierarchicalVisitor.h"
#include "HorseIR/Traversal/Visitor.h"
#include "HorseIR/Traversal/HierarchicalVisitor.h"

namespace HorseIR {

class SymbolTable;

class BlockStatement : public Statement
{
public:
	BlockStatement(const std::vector<Statement *>& statements) : m_statements(statements) {}

	BlockStatement *Clone() const override
	{
		std::vector<Statement *> statements;
		for (const auto& statement : m_statements)
		{
			statements.push_back(statement->Clone());
		}
		return new BlockStatement(statements);
	}

	const std::vector<Statement *>& GetStatements() const { return m_statements; }
	void SetStatements(const std::vector<Statement *>& statements) { m_statements = statements; }

	SymbolTable *GetSymbolTable() const { return m_symbolTable; }
	void SetSymbolTable(SymbolTable *symbolTable) { m_symbolTable = symbolTable; }

	void Accept(Visitor &visitor) override { visitor.Visit(this); }
	void Accept(ConstVisitor &visitor) const override { visitor.Visit(this); }

	void Accept(HierarchicalVisitor &visitor) override
	{
		if (visitor.VisitIn(this))
		{
			for (auto& statement : m_statements)
			{
				statement->Accept(visitor);
			}
		}
		visitor.VisitOut(this);
	}

	void Accept(ConstHierarchicalVisitor &visitor) const override
	{
		if (visitor.VisitIn(this))
		{
			for (const auto& statement : m_statements)
			{
				statement->Accept(visitor);
			}
		}
		visitor.VisitOut(this);
	}

protected:
	std::vector<Statement *> m_statements;

	SymbolTable *m_symbolTable = nullptr;
};

}
