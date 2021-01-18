#pragma once

#include "PTX/Analysis/Framework/FlowAnalysis.h"

#include "PTX/Analysis/ControlFlow/ControlFlowGraph.h"

namespace PTX {
namespace Analysis {

template<class F>
class ForwardAnalysis : public FlowAnalysis<F>
{
public:
	void TraverseFunction(const FunctionDefinition<VoidType> *function, const F& initialFlow) override
	{
		const auto cfg = function->GetControlFlowGraph();
		const auto temporaryFlow = this->TemporaryFlow();

		// Traverse parameters

		for (const auto parameter : function->GetParameters())
		{
			parameter->Accept(*this);
		}

		// Initialize worklist with start nodes

		for (const auto node : cfg->GetNodes())
		{
			if (cfg->GetInDegree(node) == 0)
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

				TraverseBlock(node);

				const auto newOutSet = this->GetOutSet(node);

				for (const auto successor : cfg->GetSuccessors(node))
				{
					const auto& successorInSet = this->GetInSet(successor);
					const auto mergedInSet = this->Merge(newOutSet, successorInSet);

					this->SetInSet(successor, mergedInSet);
					this->PushWorklist(successor);
				}
			}
			else
			{
				// For further iterations, save the old outset for comparison

				const auto oldOutSet = this->GetOutSet(node);

				TraverseBlock(node);

				const auto newOutSet = this->GetOutSet(node);

				// Only process the successors if the outset has changed

				if (oldOutSet != newOutSet)
				{
					for (const auto successor : cfg->GetSuccessors(node))
					{
						const auto& successorInSet = this->GetInSet(successor);
						const auto mergedInSet = this->Merge(newOutSet, successorInSet);

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

	void TraverseBlock(const BasicBlock *block) override
	{
		this->m_currentInSet = this->GetInSet(block);

		TraverseStatements(block->GetStatements());

		this->SetOutSet(block, this->m_currentOutSet);
	}

	void TraverseStatements(const std::vector<const Statement *>& statements)
	{
		for (const auto& statement : statements)
		{
			this->SetInSet(statement, this->m_currentInSet);
			statement->Accept(*this);
			this->SetOutSet(statement, this->m_currentOutSet);

			PropagateNext();
		}
	}

	void PropagateNext() override
	{
		// Copy the out set to the in set for traversing the next node

		this->m_currentInSet = this->m_currentOutSet;
	}

	void Visit(const Node *node) override
	{
		// Default action, propagate the set forward with no changes

		this->m_currentOutSet = this->m_currentInSet;
	}
};

}
}
