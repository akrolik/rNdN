#pragma once

#include "PTX/Analysis/Framework/ForwardControlAnalysis.h"
#include "PTX/Analysis/Framework/FlowAnalysis.h"

#include "PTX/Analysis/ControlFlow/ControlFlowGraph.h"

namespace PTX {
namespace Analysis {

template<class F>
class ForwardAnalysis : public FlowAnalysis<F, ForwardControlAnalysis>
{
public:
	F InitialFlow(const FunctionDefinition<VoidType> *function) const override
	{
		// Traverse parameters for initial flow

		this->m_currentInSet.clear();
		this->m_currentOutSet.clear();

		for (const auto parameter : function->GetParameters())
		{
			parameter->Accept(*this);
			PropagateNext();
		}

		return this->m_currentOutSet;
	}

	void TraverseBlock(const BasicBlock *block) override
	{
		TraverseStatements(block->GetStatements());
	}

	void TraverseStatements(const std::vector<const Statement *>& statements) override
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
