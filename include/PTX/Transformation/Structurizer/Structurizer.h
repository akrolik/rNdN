#pragma once

#include "PTX/Analysis/ControlFlow/StructuredGraph/StructuredGraph.h"

#include "PTX/Analysis/Dominator/DominatorAnalysis.h"
#include "PTX/Analysis/Dominator/PostDominatorAnalysis.h"

#include "PTX/Tree/Tree.h"

namespace PTX {
namespace Transformation {

class Structurizer
{
public:
	Structurizer(const Analysis::DominatorAnalysis& dominators, const Analysis::PostDominatorAnalysis& postDominators)
		: m_dominators(dominators), m_postDominators(postDominators) {}

	// Structurizer algorithm

	Analysis::StructureNode *Structurize(const FunctionDefinition<VoidType> *function);
	Analysis::StructureNode *Structurize(const Analysis::ControlFlowGraph *cfg, BasicBlock *block, bool skipLoop = false);

private:
	// Reconvergence structures

	struct Context
	{
		virtual ~Context() {};
	};

	struct LoopContext : public Context
	{
	public:
		LoopContext(const BasicBlock *header, const BasicBlock *latch, const BasicBlock *exit)
			: m_header(header), m_latch(latch), m_exit(exit) {}

		const BasicBlock *GetHeader() const { return m_header; }
		const BasicBlock *GetLatch() const { return m_latch; }
		const BasicBlock *GetExit() const { return m_exit; }

	private:
		const BasicBlock *m_header = nullptr;
		const BasicBlock *m_latch = nullptr;
		const BasicBlock *m_exit = nullptr;
	};

	struct BranchContext : public Context
	{
	public:
		BranchContext(const BasicBlock *reconvergence) : m_reconvergence(reconvergence) {}

		const BasicBlock *GetReconvergence() const { return m_reconvergence; }

	private:
		const BasicBlock *m_reconvergence = nullptr;
	};

	std::unordered_set<const BasicBlock *> m_processedNodes;
	std::stack<const Context *> m_reconvergenceStack;

	// Analyses

	const Analysis::DominatorAnalysis& m_dominators;
	const Analysis::PostDominatorAnalysis& m_postDominators;
};

}
}