#include "PTX/Analysis/Dominator/DominatorAnalysis.h"

namespace PTX {
namespace Analysis {

void DominatorAnalysis::TraverseBlock(const BasicBlock *block)
{
	// Add current block to dominators

	m_currentSet.insert(block);
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

std::unordered_set<const BasicBlock *> DominatorAnalysis::GetDominators(const BasicBlock *block) const
{
	const auto& set = this->GetOutSet(block);
	return { std::begin(set), std::end(set) };
}

std::unordered_set<const BasicBlock *> DominatorAnalysis::GetStrictDominators(const BasicBlock *block) const
{
	auto dominators = GetDominators(block);
	dominators.erase(block);
	return dominators;
}

const BasicBlock *DominatorAnalysis::GetImmediateDominator(const BasicBlock *block) const
{
	const auto& strictDominators = this->GetOutSet(block);
	for (const auto node1 : strictDominators)
	{
		// Strict dominators

		if (node1 == block)
		{
			continue;
		}

		// Check that this node dominates all other strict dominators

		auto dominatesAll = true;
		for (const auto node2 : strictDominators)
		{
			// Strict dominators

			if (node2 == block || node1 == node2)
			{
				continue;
			}

			const auto& dominators2 = this->GetOutSet(node2);
			if (dominators2.find(node1) != dominators2.end())
			{
				dominatesAll = false;
				break;
			}
		}

		// If all nodes dominated, this is our strict dominator

		if (dominatesAll)
		{
			return node1;
		}
	}
	return nullptr;
}

bool DominatorAnalysis::IsDominated(const BasicBlock *block, const BasicBlock *dominator) const
{
	const auto& blockDominators = this->GetOutSet(block);
	return (blockDominators.find(dominator) != blockDominators.end());
}

}
}
