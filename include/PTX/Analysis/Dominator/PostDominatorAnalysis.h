#pragma once

#include "PTX/Analysis/Framework/BackwardControlAnalysis.h"

#include "Analysis/FlowValue.h"

namespace PTX {
namespace Analysis {

struct PostDominatorAnalysisValue : ::Analysis::PointerValue<BasicBlock>
{
	using Type = BasicBlock;
	using ::Analysis::PointerValue<Type>::Equals;

	struct Hash
	{
		std::size_t operator()(const Type *val) const
		{
			return std::hash<const Type *>()(val);
		}
	};

	static void Print(std::ostream& os, const Type *val)
	{
		os << val->GetLabel()->GetName();
	}
};


using PostDominatorAnalysisProperties = ::Analysis::Set<PostDominatorAnalysisValue>; 

class PostDominatorAnalysis : public BackwardControlAnalysis<PostDominatorAnalysisProperties>
{
public:
	using Properties = PostDominatorAnalysisProperties;
	using BackwardControlAnalysis<PostDominatorAnalysisProperties>::BackwardControlAnalysis;

	// Accessors

	std::unordered_set<const BasicBlock *> GetPostDominators(const BasicBlock *block) const
	{
		const auto& set = this->GetInSet(block);
		return { std::begin(set), std::end(set) };
	}

	std::unordered_set<const BasicBlock *> GetStrictPostDominators(const BasicBlock *block) const
	{
		auto dominators = GetPostDominators(block);
		dominators.erase(block);
		return dominators;
	}

	const BasicBlock *GetImmediatePostDominator(const BasicBlock *block) const
	{
		const auto strictPostDominators = GetStrictPostDominators(block);
		for (const auto node1 : strictPostDominators)
		{
			// Check that this node post-dominates all other strict post-dominators

			auto postDominatesAll = true;
			for (const auto node2 : strictPostDominators)
			{
				const auto strictPostDominators2 = GetStrictPostDominators(node2);
				if (strictPostDominators2.find(node1) != strictPostDominators2.end())
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

	// Visitors

	void TraverseBlock(const BasicBlock *block) override;

	// Flow

	Properties InitialFlow(const FunctionDefinition<VoidType> *function) const override;
	Properties TemporaryFlow(const FunctionDefinition<VoidType> *function) const override;

	Properties Merge(const Properties& s1, const Properties& s2) const override;

	std::string Name() const override { return "Post-dominators"; }
};

}
}
