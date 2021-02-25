#include "PTX/Analysis/Dominator/DominatorAnalysis.h"

namespace PTX {
namespace Analysis {

void DominatorAnalysis::TraverseBlock(const BasicBlock *block)
{
	// Copy input to output directly

	m_currentOutSet = m_currentInSet;
	m_currentOutSet.insert(block);
}

DominatorAnalysis::Properties DominatorAnalysis::InitialFlow(const FunctionDefinition<VoidType> *function) const
{
	return {};
}

DominatorAnalysis::Properties DominatorAnalysis::TemporaryFlow(const FunctionDefinition<VoidType> *function) const
{
	// All nodes included in temporary flow

	const auto cfg = function->GetControlFlowGraph();
	const auto nodes = cfg->GetNodes();

	Properties temporary(std::begin(nodes), std::end(nodes));
	return temporary;
}

DominatorAnalysis::Properties DominatorAnalysis::Merge(const Properties& s1, const Properties& s2) const
{
	// Merge the sets using intersection

	Properties outSet;
	for (const auto& block : s1)
	{
		if (s2.find(block) != s2.end())
		{
			outSet.insert(block);
		}
	}
	return outSet;
}

}
}
