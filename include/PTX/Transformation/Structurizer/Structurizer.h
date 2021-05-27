#pragma once

#include <stack>
#include <vector>

#include "PTX/Analysis/ControlFlow/StructuredGraph/StructuredGraph.h"

#include "PTX/Analysis/Dominator/DominatorAnalysis.h"
#include "PTX/Analysis/Dominator/PostDominatorAnalysis.h"

#include "PTX/Tree/Tree.h"

#include "Libraries/robin_hood.h"

namespace PTX {
namespace Transformation {

class Structurizer
{
public:
	Structurizer(const Analysis::DominatorAnalysis& dominators, const Analysis::PostDominatorAnalysis& postDominators)
		: m_dominators(dominators), m_postDominators(postDominators) {}

	// Structurizer algorithm

	Analysis::StructureNode *Structurize(FunctionDefinition<VoidType> *function);
	Analysis::StructureNode *Structurize(const Analysis::ControlFlowGraph *cfg, BasicBlock *block, bool skipLoop = false);

private:
	robin_hood::unordered_set<BasicBlock *> GetLoopBlocks(const Analysis::ControlFlowGraph *cfg, BasicBlock *header, BasicBlock *latch) const;
	BasicBlock *GetLoopExit(BasicBlock *header, const robin_hood::unordered_set<BasicBlock *>& loopBlocks) const;

	[[noreturn]] void Error(const std::string& message, const BasicBlock *block);

	// Reconvergence structures

	struct Context
	{
		enum Kind {
			Function,
			Loop,
			Branch
		};

		Context(Kind kind) : m_kind(kind) {}

		Kind GetKind() const { return m_kind; }
		Kind m_kind;

		virtual ~Context() {};
	};

	struct LoopContext : public Context
	{
	public:
		LoopContext(BasicBlock *header, BasicBlock *latch, BasicBlock *exit, const robin_hood::unordered_set<BasicBlock *>& loopBlocks)
			: Context(Kind::Loop), m_header(header), m_latch(latch), m_exit(exit), m_loopBlocks(loopBlocks) {}

		BasicBlock *GetHeader() const { return m_header; }
		BasicBlock *GetLatch() const { return m_latch; }
		BasicBlock *GetExit() const { return m_exit; }

		const robin_hood::unordered_set<BasicBlock *>& GetLoopBlocks() const { return m_loopBlocks; }
		bool ContainsBlock(BasicBlock *block) const { return m_loopBlocks.find(block) != m_loopBlocks.end(); }

	private:
		BasicBlock *m_header = nullptr;
		BasicBlock *m_latch = nullptr;
		BasicBlock *m_exit = nullptr;

		robin_hood::unordered_set<BasicBlock *> m_loopBlocks;
	};

	struct BranchContext : public Context
	{
	public:
		BranchContext(BasicBlock *reconvergence) : Context(Context::Branch), m_reconvergence(reconvergence) {}

		BasicBlock *GetReconvergence() const { return m_reconvergence; }

	private:
		BasicBlock *m_reconvergence = nullptr;
	};

	robin_hood::unordered_set<BasicBlock *> m_processedNodes;
	std::stack<const Context *> m_reconvergenceStack;
	robin_hood::unordered_set<Analysis::ExitStructure *> m_exitStructures;
	Analysis::StructureNode *m_latchStructure = nullptr;

	// Analyses

	const Analysis::DominatorAnalysis& m_dominators;
	const Analysis::PostDominatorAnalysis& m_postDominators;
};

}
}
