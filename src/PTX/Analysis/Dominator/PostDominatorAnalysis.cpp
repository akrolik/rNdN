#include "PTX/Analysis/Dominator/PostDominatorAnalysis.h"

namespace PTX {
namespace Analysis {

void PostDominatorAnalysis::TraverseBlock(const BasicBlock *block)
{
	// Add current set to dominators

	m_currentSet.insert(block);
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

std::unordered_set<const BasicBlock *> PostDominatorAnalysis::GetPostDominators(const BasicBlock *block) const
{
	const auto& set = this->GetInSet(block);
	return { std::begin(set), std::end(set) };
}

std::unordered_set<const BasicBlock *> PostDominatorAnalysis::GetStrictPostDominators(const BasicBlock *block) const
{
	auto dominators = GetPostDominators(block);
	dominators.erase(block);
	return dominators;
}

const BasicBlock *PostDominatorAnalysis::GetImmediatePostDominator(const BasicBlock *block) const
{
	const auto& postDominators = this->GetInSet(block);
	for (const auto node1 : postDominators)
	{
		// Strict post dominators

		if (node1 == block)
		{
			continue;
		}

		// Check that this node post-dominates all other strict post-dominators

		auto postDominatesAll = true;
		for (const auto node2 : postDominators)
		{
			// Strict post dominators

			if (node2 == block || node1 == node2)
			{
				continue;
			}

			const auto& postDominators2 = this->GetInSet(node2);
			if (postDominators2.find(node1) != postDominators2.end())
			{
				postDominatesAll = false;
				break;
			}
		}
		// If all nodes post-dominated, this is our immediate post-dominator

		if (postDominatesAll)
		{
			return node1;
		}
	}
	return nullptr;
}

bool PostDominatorAnalysis::IsPostDominated(const BasicBlock *block, const BasicBlock *postDominator) const
{
	const auto& postDominators = this->GetInSet(block);
	return (postDominators.find(postDominator) != postDominators.end());
}

}
}
