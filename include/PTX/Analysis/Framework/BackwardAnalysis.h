#pragma once

#include "PTX/Analysis/Framework/FlowAnalysis.h"

#include "PTX/Analysis/ControlFlow/ControlFlowGraph.h"

namespace PTX {
namespace Analysis {

template<class F>
class BackwardAnalysis : public FlowAnalysis<F>
{
public:
	void TraverseFunction(const FunctionDefinition<VoidType> *function, const F& initialFlow) override
	{
		const auto cfg = function->GetControlFlowGraph();
		const auto temporaryFlow = this->TemporaryFlow();

		// Initialize worklist with end nodes

		for (const auto node : cfg->GetNodes())
		{
			if (cfg->GetOutDegree(node) == 0)
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

				TraverseBlock(node);

				const auto newInSet = this->GetInSet(node);

				for (const auto predecessor : cfg->GetPredecessors(node))
				{
					const auto& predecessorOutSet = this->GetOutSet(predecessor);
					const auto mergedOutSet = this->Merge(newInSet, predecessorOutSet);

					this->SetOutSet(predecessor, mergedOutSet);
					this->PushWorklist(predecessor);
				}
			}
			else
			{
				// For further iterations, save the old inset for comparison

				const auto oldInSet = this->GetInSet(node);

				TraverseBlock(node);

				const auto newInSet = this->GetInSet(node);

				// Only process the predecessors if the inset has changed

				if (oldInSet != newInSet)
				{
					for (const auto predecessor : cfg->GetPredecessors(node))
					{
						const auto& predecessorOutSet = this->GetOutSet(predecessor);
						const auto mergedOutSet = this->Merge(newInSet, predecessorOutSet);

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

	void TraverseBlock(const BasicBlock *block) override
	{
		this->m_currentOutSet = this->GetOutSet(block);

		TraverseStatements(block->GetStatements());

		this->SetInSet(block, this->m_currentInSet);
	}

	void TraverseStatements(const std::vector<const Statement *>& statements)
	{
		for (auto it = statements.rbegin(); it != statements.rend(); ++it)
		{
			auto statement = *it;

			this->SetOutSet(statement, this->m_currentOutSet);
			statement->Accept(*this);
			this->SetInSet(statement, this->m_currentInSet);

			this->PropagateNext();
		}
	}

	void PropagateNext() override
	{
		// Copy the in set to the out set for traversing the next node

		this->m_currentOutSet = this->m_currentInSet;
	}

	void Visit(const Node *node) override
	{
		// Default action, propagate the set forward with no changes

		this->m_currentInSet = this->m_currentOutSet;
	}
};

}
}
