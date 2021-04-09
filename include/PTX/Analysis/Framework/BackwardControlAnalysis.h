#pragma once

#include "PTX/Analysis/Framework/ControlFlowAnalysis.h"

#include "PTX/Analysis/ControlFlow/ControlFlowGraph.h"

namespace PTX {
namespace Analysis {

template<class F>
class BackwardControlAnalysis : public ControlFlowAnalysis<F>
{
public:
	void TraverseFunction(const FunctionDefinition<VoidType> *function, const F& initialFlow) override
	{
		const auto cfg = function->GetControlFlowGraph();
		const auto temporaryFlow = this->TemporaryFlow(function);

		// Initialize worklist with end nodes (multiple possible)

		for (const auto node : cfg->GetExitNodes())
		{
			this->PushWorklist(node);
			this->SetOutSet(node, initialFlow);
		}

		// Traverse worklist in order

		while (!this->IsEmptyWorklist())
		{
			const auto node = this->PopWorklist();

			// Process the current node

			this->m_currentSet = this->GetOutSet(node);
			TraverseBlock(node);

			if (this->CollectInSets())
			{
				this->SetInSet(node, this->m_currentSet);
			}

			// Process the predecessors

			for (const auto predecessor : cfg->GetPredecessors(node))
			{
				if (this->ContainsOutSet(predecessor))
				{
					const auto& predecessorOutSet = this->GetOutSet(predecessor);
					auto mergedOutSet = this->Merge(this->m_currentSet, predecessorOutSet);

					// Proccess changed predecessors

					if (mergedOutSet != predecessorOutSet)
					{
						this->SetOutSet(predecessor, std::move(mergedOutSet));
						this->PushWorklist(predecessor);
					}
				}
				else
				{
					// Proccess new predecessors

					auto mergedOutSet = this->Merge(this->m_currentSet, temporaryFlow);

					this->SetOutSetMove(predecessor, std::move(mergedOutSet));
					this->PushWorklist(predecessor);
				}
			}
		}
	}

	virtual void TraverseBlock(const BasicBlock *block) = 0;

protected:
	void InitializeWorklist(const FunctionDefinition<VoidType> *function) override
	{
		const auto cfg = function->GetControlFlowGraph();
		const auto entry = cfg->GetEntryNode();
		auto index = 0;

		cfg->DFS(entry, [&](BasicBlock *block)
		{
			this->m_blockOrder[block] = index;
			index++;

			return false;
		}, Utils::Graph<BasicBlock *>::Traversal::Postorder);
	}
};

}
}
