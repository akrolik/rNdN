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

	robin_hood::unordered_set<const BasicBlock *> GetDominators(const BasicBlock *block) const;
	robin_hood::unordered_set<const BasicBlock *> GetStrictDominators(const BasicBlock *block) const;
	const BasicBlock *GetImmediateDominator(const BasicBlock *block) const;

	bool IsDominated(const BasicBlock *block, const BasicBlock *dominator) const;

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
