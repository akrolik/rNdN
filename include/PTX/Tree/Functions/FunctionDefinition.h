#pragma once

#include <unordered_set>

#include "PTX/Tree/Functions/FunctionDeclaration.h"

#include "PTX/Tree/BasicBlock.h"
#include "PTX/Tree/Statements/StatementList.h"
#include "PTX/Tree/Statements/Statement.h"

#include "PTX/Analysis/ControlFlow/ControlFlowGraph.h"

namespace PTX {

class VoidType;

template<class R>
class FunctionDefinition : public FunctionDeclaration<R>, public StatementList
{
public:
	// Control-flow graph

	const Analysis::ControlFlowGraph *GetControlFlowGraph() const { return m_cfg; }
	Analysis::ControlFlowGraph *GetControlFlowGraph() { return m_cfg; }
	void SetControlFlowGraph(Analysis::ControlFlowGraph *cfg) { m_cfg = cfg; }

	void InvalidateStatements() { m_statements.clear(); }

	// Basic blocks

	std::unordered_set<const BasicBlock *> GetBasicBlocks() const
	{
		return { std::begin(m_basicBlocks), std::end(m_basicBlocks) };
	}
	std::unordered_set<BasicBlock *>& GetBasicBlocks() { return m_basicBlocks; }
	void SetBasicBlocks(const std::unordered_set<BasicBlock *>& basicBlocks) { m_basicBlocks = basicBlocks; }

	// Formatting

	json ToJSON() const override
	{
		json j = FunctionDeclaration<R>::ToJSON();
		if (m_cfg == nullptr)
		{
			j["statements"] = StatementList::ToJSON();
		}
		else
		{
			for (const auto& block : m_basicBlocks)
			{
				j["basic_blocks"].push_back(block->ToJSON());
			}
		}
		return j;
	}

	// Visitors

	void Accept(Visitor& visitor) override
	{
		if constexpr(std::is_same<R, VoidType>::value)
		{
			visitor.Visit(this);
		}
	}

	void Accept(ConstVisitor& visitor) const override
	{
		if constexpr(std::is_same<R, VoidType>::value)
		{
			visitor.Visit(this);
		}
	}

	void Accept(HierarchicalVisitor& visitor) override
	{
		if constexpr(std::is_same<R, VoidType>::value)
		{
			if (visitor.VisitIn(this))
			{
				for (auto& parameter : FunctionDeclaration<R>::m_parameters)
				{
					parameter->Accept(visitor);
				}

				if (m_cfg != nullptr)
				{
					m_cfg->LinearOrdering([&](Analysis::ControlFlowNode& block)
					{
						block->Accept(visitor);
					});
				}
				else
				{
					for (auto& statement : m_statements)
					{
						statement->Accept(visitor);
					}
				}
			}
			visitor.VisitOut(this);
		}
	}

	void Accept(ConstHierarchicalVisitor& visitor) const override
	{
		if constexpr(std::is_same<R, VoidType>::value)
		{
			if (visitor.VisitIn(this))
			{
				for (const auto& parameter : FunctionDeclaration<R>::m_parameters)
				{
					parameter->Accept(visitor);
				}
				if (m_cfg != nullptr)
				{
					m_cfg->LinearOrdering([&](const Analysis::ControlFlowNode& block)
					{
						block->Accept(visitor);
					});
				}
				else
				{
					for (const auto& statement : m_statements)
					{
						statement->Accept(visitor);
					}
				}
			}
			visitor.VisitOut(this);
		}
	}

protected:
	std::unordered_set<BasicBlock *> m_basicBlocks;
	Analysis::ControlFlowGraph *m_cfg = nullptr;
};

}
