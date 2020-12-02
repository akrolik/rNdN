#pragma once

#include "PTX/Traversal/ConstHierarchicalVisitor.h"

#include "PTX/Tree/Tree.h"
#include "PTX/Analysis/ControlFlow/ControlFlowGraph.h"

namespace PTX {
namespace Analysis {

class ControlFlowAccumulator : public ConstHierarchicalVisitor
{
public:
	// API

	void Analyze(const FunctionDefinition<VoidType> *function);

	ControlFlowGraph *GetGraph() const { return m_graph; }
	const std::unordered_map<const Statement *, const BasicBlock *>& GetBlockMap() const { return m_blockMap; }

	// Statements

	bool VisitIn(const Statement *statement) override;
	bool VisitIn(const Label *label) override;
	
private:
	BasicBlock *m_currentBlock = nullptr;
	ControlFlowGraph *m_graph = nullptr;

	std::unordered_map<const Statement *, const BasicBlock *> m_blockMap;
};

class ControlFlowBuilder : public ConstHierarchicalVisitor
{
public:
	// API

	void Analyze(const FunctionDefinition<VoidType> *function);
	ControlFlowGraph *GetGraph() const { return m_graph; }
	
	// Statements

	bool VisitIn(const Statement *statement) override;
	bool VisitIn(const InstructionStatement *statement) override;
	bool VisitIn(const Label *label) override;

private:
	const BasicBlock *m_previousBlock = nullptr;

	ControlFlowGraph *m_graph = nullptr;
	std::unordered_map<const Statement *, const BasicBlock *> m_blockMap;
};

}
}
