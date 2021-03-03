#pragma once

#include "PTX/Analysis/Framework/ForwardControlAnalysis.h"

#include "Analysis/FlowValue.h"

namespace PTX {
namespace Analysis {

struct DominatorAnalysisValue : ::Analysis::PointerValue<BasicBlock>
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


using DominatorAnalysisProperties = ::Analysis::Set<DominatorAnalysisValue>; 

class DominatorAnalysis : public ForwardControlAnalysis<DominatorAnalysisProperties>
{
public:
	using Properties = DominatorAnalysisProperties;
	using ForwardControlAnalysis<DominatorAnalysisProperties>::ForwardControlAnalysis;

	// Accessors

	std::unordered_set<const BasicBlock *> GetDominators(const BasicBlock *block) const
	{
		const auto& set = this->GetOutSet(block);
		return { std::begin(set), std::end(set) };
	}

	std::unordered_set<const BasicBlock *> GetStrictDominators(const BasicBlock *block) const
	{
		auto dominators = GetDominators(block);
		dominators.erase(block);
		return dominators;
	}

	const BasicBlock *GetImmediateDominator(const BasicBlock *block) const
	{
		const auto strictDominators = GetStrictDominators(block);
		for (const auto node1 : strictDominators)
		{
			// Check that this node dominates all other strict dominators

			auto dominatesAll = true;
			for (const auto node2 : strictDominators)
			{
				const auto strictDominators2 = GetStrictDominators(node2);
				if (strictDominators2.find(node1) != strictDominators2.end())
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

	// Visitors

	void TraverseBlock(const BasicBlock *block) override;

	// Flow

	Properties InitialFlow(const FunctionDefinition<VoidType> *function) const override;
	Properties TemporaryFlow(const FunctionDefinition<VoidType> *function) const override;

	Properties Merge(const Properties& s1, const Properties& s2) const override;

	std::string Name() const override { return "Dominators"; }
};

}
}
