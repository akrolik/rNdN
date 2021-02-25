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

		for (const auto node : cfg->GetNodes())
		{
			if (cfg->IsExitNode(node))
			{
				this->PushWorklist(node);
				this->SetOutSet(node, initialFlow);
			}
			else
			{
				this->SetOutSet(node, temporaryFlow);
			}
		}

		// Traverse worklist in order

		while (!this->IsEmptyWorklist())
		{
			const auto node = this->PopWorklist();

			if (!this->ContainsInSet(node))
			{
				// If this is the first iteration, travese the block and all predecessors

				this->m_currentOutSet = this->GetOutSet(node);
				TraverseBlock(node);
				this->SetInSet(node, this->m_currentInSet);

				// Propagate to all successors

				for (const auto predecessor : cfg->GetPredecessors(node))
				{
					const auto& predecessorOutSet = this->GetOutSet(predecessor);
					const auto mergedOutSet = this->Merge(this->m_currentInSet, predecessorOutSet);

					this->SetOutSet(predecessor, mergedOutSet);
					this->PushWorklist(predecessor);
				}
			}
			else
			{
				// For further iterations, save the old inset for comparison

				const auto oldInSet = this->GetInSet(node);

				this->m_currentOutSet = this->GetOutSet(node);
				TraverseBlock(node);
				this->SetInSet(node, this->m_currentInSet);

				// Only process the predecessors if the inset has changed

				if (oldInSet != this->m_currentInSet)
				{
					for (const auto predecessor : cfg->GetPredecessors(node))
					{
						const auto& predecessorOutSet = this->GetOutSet(predecessor);
						const auto mergedOutSet = this->Merge(this->m_currentInSet, predecessorOutSet);

						// Proccess changed predecessors

						if (mergedOutSet != predecessorOutSet)
						{
							this->SetOutSet(predecessor, mergedOutSet);
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
