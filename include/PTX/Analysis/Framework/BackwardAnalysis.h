#pragma once

#include "PTX/Analysis/Framework/BackwardControlAnalysis.h"
#include "PTX/Analysis/Framework/FlowAnalysis.h"

#include "PTX/Analysis/ControlFlow/ControlFlowGraph.h"

namespace PTX {
namespace Analysis {

template<class F>
class BackwardAnalysis : public FlowAnalysis<F, BackwardControlAnalysis>
{
public:
	void TraverseStatements(const std::vector<const Statement *>& statements) override
	{
		for (auto it = statements.rbegin(); it != statements.rend(); ++it)
		{
			auto statement = *it;

			this->SetOutSet(statement, this->m_currentSet);
			statement->Accept(*this);
			this->SetInSet(statement, this->m_currentSet);
		}
	}

	void Visit(const Node *node) override
	{
		// Default action, propagate the set forward with no changes
	}
};

}
}
