#pragma once

#include "PTX/Traversal/HierarchicalVisitor.h"

#include "PTX/Tree/Tree.h"
#include "PTX/Analysis/ControlFlow/ControlFlowGraph.h"

namespace PTX {
namespace Analysis {

class ControlFlowAccumulator : public HierarchicalVisitor
{
public:
	// API

	ControlFlowGraph *Analyze(FunctionDefinition<VoidType> *function);

	const std::unordered_map<const Label *, const BasicBlock *>& GetLabelMap() const { return m_labelMap; }
	const std::unordered_map<const Statement *, const BasicBlock *>& GetStatementMap() const { return m_statementMap; }

	// Statements

	bool VisitIn(Statement *statement) override;
	bool VisitIn(LabelStatement *statement) override;
	
private:
	unsigned int m_index = 0u;
	BasicBlock *m_currentBlock = nullptr;

	ControlFlowGraph *m_graph = nullptr;
	std::unordered_map<const Label *, const BasicBlock *> m_labelMap;
	std::unordered_map<const Statement *, const BasicBlock *> m_statementMap;
};

class ControlFlowBuilder : public HierarchicalVisitor
{
public:
	// API

	ControlFlowGraph *Analyze(FunctionDefinition<VoidType> *function);
	
	// Statements

	bool VisitIn(Statement *statement) override;
	bool VisitIn(InstructionStatement *statement) override;
	bool VisitIn(LabelStatement *statement) override;

private:
	const BasicBlock *m_previousBlock = nullptr;

	ControlFlowGraph *m_graph = nullptr;
	std::unordered_map<const Label *, const BasicBlock *> m_labelMap;
	std::unordered_map<const Statement *, const BasicBlock *> m_statementMap;
};

}
}
