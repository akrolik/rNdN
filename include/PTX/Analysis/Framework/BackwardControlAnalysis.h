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

			if (!this->ContainsInSet(node))
			{
				// If this is the first iteration, travese the block and all predecessors

				this->m_currentSet = this->GetOutSet(node);
				TraverseBlock(node);
				this->SetInSet(node, this->m_currentSet);

				// Propagate to all successors

				for (const auto predecessor : cfg->GetPredecessors(node))
				{
					if (this->ContainsOutSet(predecessor))
					{
						const auto& predecessorOutSet = this->GetOutSet(predecessor);
						this->SetOutSet(predecessor, this->Merge(this->m_currentSet, predecessorOutSet));
					}
					else
					{
						this->SetOutSet(predecessor, this->Merge(this->m_currentSet, temporaryFlow));
					}
					this->PushWorklist(predecessor);
				}
			}
			else
			{
				// For further iterations, only process the predecessors if the inset has changed

				this->m_currentSet = this->GetOutSet(node);
				TraverseBlock(node);

				if (this->GetInSet(node) != this->m_currentSet)
				{
					this->SetInSet(node, this->m_currentSet);

					for (const auto predecessor : cfg->GetPredecessors(node))
					{
						if (this->ContainsOutSet(predecessor))
						{
							const auto& predecessorOutSet = this->GetOutSet(predecessor);
							const auto mergedOutSet = this->Merge(this->m_currentSet, predecessorOutSet);

							// Proccess changed predecessors

							if (mergedOutSet != predecessorOutSet)
							{
								this->SetOutSet(predecessor, mergedOutSet);
								this->PushWorklist(predecessor);
							}
						}
						else
						{
							// Proccess changed predecessors

							this->SetOutSet(predecessor, this->Merge(this->m_currentSet, temporaryFlow));
							this->PushWorklist(predecessor);
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
