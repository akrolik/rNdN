#pragma once

#include "PTX/Traversal/HierarchicalVisitor.h"

#include "PTX/Tree/Tree.h"
#include "PTX/Analysis/ControlFlow/ControlFlowGraph.h"

#include "Libraries/robin_hood.h"

namespace PTX {
namespace Analysis {

class ControlFlowAccumulator : public HierarchicalVisitor
{
public:
	// API

	ControlFlowGraph *Analyze(FunctionDefinition<VoidType> *function);

	const robin_hood::unordered_map<const Label *, BasicBlock *>& GetLabelMap() const { return m_labelMap; }
	const robin_hood::unordered_map<const Statement *, BasicBlock *>& GetStatementMap() const { return m_statementMap; }

	// Statements

	bool VisitIn(Statement *statement) override;
	bool VisitIn(ReturnInstruction *instruction) override;
	bool VisitIn(BranchInstruction *instruction) override;
	bool VisitIn(PredicatedInstruction *instruction) override;
	bool VisitIn(LabelStatement *statement) override;
	
private:
	BasicBlock *CreateBlock();
	BasicBlock *CreateBlock(const std::string& name);

	bool m_entry = true;
	unsigned int m_index = 0u;
	BasicBlock *m_currentBlock = nullptr;

	ControlFlowGraph *m_graph = nullptr;
	robin_hood::unordered_map<const Label *, BasicBlock *> m_labelMap;
	robin_hood::unordered_map<const Statement *, BasicBlock *> m_statementMap;
};

class ControlFlowBuilder : public HierarchicalVisitor
{
public:
	// API

	ControlFlowGraph *Analyze(FunctionDefinition<VoidType> *function);
	
	// Statements

	bool VisitIn(Statement *statement) override;
	bool VisitIn(BranchInstruction *instruction) override;
	bool VisitIn(PredicatedInstruction *instruction) override;
	bool VisitIn(LabelStatement *statement) override;

private:
	BasicBlock *m_previousBlock = nullptr;

	ControlFlowGraph *m_graph = nullptr;
	robin_hood::unordered_map<const Label *, BasicBlock *> m_labelMap;
	robin_hood::unordered_map<const Statement *, BasicBlock *> m_statementMap;
};

}
}
