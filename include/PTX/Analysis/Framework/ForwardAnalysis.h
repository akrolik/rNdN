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
	void TraverseStatements(const std::vector<const Statement *>& statements) override
	{
		for (const auto& statement : statements)
		{
			this->SetInSet(statement, this->m_currentSet);
			statement->Accept(*this);
			this->SetOutSet(statement, this->m_currentSet);
		}
	}

	void Visit(const Node *node) override
	{
		// Default action, propagate the set forward with no changes
	}
};

}
}
