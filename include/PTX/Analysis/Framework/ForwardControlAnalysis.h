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

		auto entry = cfg->GetEntryNode();

		this->PushWorklist(entry);
		this->SetInSet(entry, initialFlow);

		// Traverse worklist in order

		while (!this->IsEmptyWorklist())
		{
			const auto node = this->PopWorklist();
			if (!this->ContainsOutSet(node))
			{
				// If this is the first iteration, travese the block and all successors

				this->m_currentSet = this->GetInSet(node);
				TraverseBlock(node);
				this->SetOutSet(node, this->m_currentSet);

				// Propagate to all successors

				for (const auto successor : cfg->GetSuccessors(node))
				{
					if (this->ContainsInSet(successor))
					{
						const auto& successorInSet = this->GetInSet(successor);
						this->SetInSet(successor, this->Merge(this->m_currentSet, successorInSet));
					}
					else
					{
						this->SetInSet(successor, this->Merge(this->m_currentSet, temporaryFlow));
					}
					this->PushWorklist(successor);
				}
			}
			else
			{
				// For further iterations, save the old outset for comparison

				this->m_currentSet = this->GetInSet(node);
				TraverseBlock(node);

				// Only process the successors if the outset has changed

				if (this->GetOutSet(node) != this->m_currentSet)
				{
					this->SetOutSet(node, this->m_currentSet);

					for (const auto successor : cfg->GetSuccessors(node))
					{
						if (this->ContainsInSet(successor))
						{
							const auto& successorInSet = this->GetInSet(successor);
							const auto mergedInSet = this->Merge(this->m_currentSet, successorInSet);

							// Proccess changed successors

							if (mergedInSet != successorInSet)
							{
								this->SetInSet(successor, mergedInSet);
								this->PushWorklist(successor);
							}
						}
						else
						{
							// Proccess changed successors

							this->SetInSet(successor, this->Merge(this->m_currentSet, temporaryFlow));
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
