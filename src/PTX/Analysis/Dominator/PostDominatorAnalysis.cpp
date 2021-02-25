#include "PTX/Analysis/Dominator/PostDominatorAnalysis.h"

namespace PTX {
namespace Analysis {

void PostDominatorAnalysis::TraverseBlock(const BasicBlock *block)
{
	// Copy input to output directly

	m_currentInSet = m_currentOutSet;
	m_currentInSet.insert(block);
}

PostDominatorAnalysis::Properties PostDominatorAnalysis::InitialFlow(const FunctionDefinition<VoidType> *function) const
{
	return {};
}

PostDominatorAnalysis::Properties PostDominatorAnalysis::TemporaryFlow(const FunctionDefinition<VoidType> *function) const
{
	// All nodes included in temporary flow

	const auto cfg = function->GetControlFlowGraph();
	const auto nodes = cfg->GetNodes();

	Properties temporary(std::begin(nodes), std::end(nodes));
	return temporary;
}

PostDominatorAnalysis::Properties PostDominatorAnalysis::Merge(const Properties& s1, const Properties& s2) const
{
	// Merge the sets using intersection

	Properties inSet;
	for (const auto& block : s1)
	{
		if (s2.find(block) != s2.end())
		{
			inSet.insert(block);
		}
	}
	return inSet;
}

}
}
