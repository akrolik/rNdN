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

	inline const static std::string Name = "Post-dominators";
	inline const static std::string ShortName = "pdom";

	PostDominatorAnalysis() : BackwardControlAnalysis<PostDominatorAnalysisProperties>(Name, ShortName) {}

	// Accessors

	robin_hood::unordered_set<const BasicBlock *> GetPostDominators(const BasicBlock *block) const;
	robin_hood::unordered_set<const BasicBlock *> GetStrictPostDominators(const BasicBlock *block) const;
	const BasicBlock *GetImmediatePostDominator(const BasicBlock *block) const;

	bool IsPostDominated(const BasicBlock *block, const BasicBlock *postDominator) const;

	// Visitors

	void TraverseBlock(const BasicBlock *block) override;

	// Flow

	Properties InitialFlow(const FunctionDefinition<VoidType> *function) const override;
	Properties TemporaryFlow(const FunctionDefinition<VoidType> *function) const override;

	Properties Merge(const Properties& s1, const Properties& s2) const override;
};

}
}
