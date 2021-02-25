#pragma once

#include "PTX/Analysis/Framework/ControlFlowAnalysis.h"

#include "PTX/Analysis/ControlFlow/ControlFlowGraph.h"

namespace PTX {
namespace Analysis {

template<class F>
class ForwardControlAnalysis : public ControlFlowAnalysis<F>
{
public:
	void TraverseFunction(const FunctionDefinition<VoidType> *function, const F& initialFlow) override
	{
		const auto cfg = function->GetControlFlowGraph();
		const auto temporaryFlow = this->TemporaryFlow(function);

		// Initialize worklist with start nodes

		for (const auto node : cfg->GetNodes())
		{
			if (cfg->IsEntryNode(node))
			{
				this->PushWorklist(node);
				this->SetInSet(node, initialFlow);
			}
			else
			{
				this->SetInSet(node, temporaryFlow);
			}
		}

		// Traverse worklist in order

		while (!this->IsEmptyWorklist())
		{
			const auto node = this->PopWorklist();
			if (!this->ContainsOutSet(node))
			{
				// If this is the first iteration, travese the block and all successors

				this->m_currentInSet = this->GetInSet(node);
				TraverseBlock(node);
				this->SetOutSet(node, this->m_currentOutSet);

				// Propagate to all successors

				for (const auto successor : cfg->GetSuccessors(node))
				{
					const auto& successorInSet = this->GetInSet(successor);
					const auto mergedInSet = this->Merge(this->m_currentOutSet, successorInSet);

					this->SetInSet(successor, mergedInSet);
					this->PushWorklist(successor);
				}
			}
			else
			{
				// For further iterations, save the old outset for comparison

				const auto oldOutSet = this->GetOutSet(node);

				this->m_currentInSet = this->GetInSet(node);
				TraverseBlock(node);
				this->SetOutSet(node, this->m_currentOutSet);

				// Only process the successors if the outset has changed

				if (oldOutSet != this->m_currentOutSet)
				{
					for (const auto successor : cfg->GetSuccessors(node))
					{
						const auto& successorInSet = this->GetInSet(successor);
						const auto mergedInSet = this->Merge(this->m_currentOutSet, successorInSet);

						// Proccess changed successors

						if (mergedInSet != successorInSet)
						{
							this->SetInSet(successor, mergedInSet);
							this->PushWorklist(successor);
						}
					}
				}
			}
		}
	}

	virtual void TraverseBlock(const BasicBlock *block) = 0;
};

}
}
