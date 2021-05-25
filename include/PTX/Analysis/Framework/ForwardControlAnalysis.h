#pragma once

#include "PTX/Analysis/Framework/ControlFlowAnalysis.h"

#include "PTX/Analysis/ControlFlow/ControlFlowGraph.h"

namespace PTX {
namespace Analysis {

template<class F>
class ForwardControlAnalysis : public ControlFlowAnalysis<F>
{
public:
	using ControlFlowAnalysis<F>::ControlFlowAnalysis;

	void TraverseFunction(const FunctionDefinition<VoidType> *function, const F& initialFlow) override
	{
		const auto cfg = function->GetControlFlowGraph();
		const auto temporaryFlow = this->TemporaryFlow(function);

		// Initialize worklist with start nodes

		auto entry = cfg->GetEntryNode();

		this->PushWorklist(entry);
		this->SetInSet(entry, initialFlow);

		// Traverse worklist in order

		while (!this->IsEmptyWorklist())
		{
			const auto node = this->PopWorklist();

			// Process the current node

			this->m_currentSet = this->GetInSet(node);
			TraverseBlock(node);

			if (this->CollectOutSets())
			{
				this->SetOutSet(node, this->m_currentSet);
			}

			// Process the successors

			for (const auto successor : cfg->GetSuccessors(node))
			{
				if (this->ContainsInSet(successor))
				{
					const auto& successorInSet = this->GetInSet(successor);
					auto mergedInSet = this->Merge(this->m_currentSet, successorInSet);

					// Proccess changed successors

					if (mergedInSet != successorInSet)
					{
						this->SetInSet(successor, std::move(mergedInSet));
						this->PushWorklist(successor);
					}
				}
				else
				{
					// Proccess new successors

					auto mergedInSet = this->Merge(this->m_currentSet, temporaryFlow);

					this->SetInSetMove(successor, std::move(mergedInSet));
					this->PushWorklist(successor);
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
		auto index = cfg->GetNodeCount();

		cfg->DFS(entry, [&](BasicBlock *block)
		{
			this->m_blockOrder[block] = index;
			index--;

			return false;
		}, Utils::Graph<BasicBlock *>::Traversal::Postorder);
	}
};

}
}
