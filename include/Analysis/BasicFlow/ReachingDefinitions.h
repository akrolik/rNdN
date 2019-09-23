#pragma once

#include <sstream>
#include <unordered_set>

#include "HorseIR/Analysis/ForwardAnalysis.h"

#include "Analysis/Utils/SymbolObject.h"

#include "HorseIR/Tree/Tree.h"
#include "HorseIR/Utils/PrettyPrinter.h"

namespace Analysis {

struct ReachingDefinitionsValue : HorseIR::FlowAnalysisValue<std::unordered_set<const HorseIR::AssignStatement *>>
{
	using Type = std::unordered_set<const HorseIR::AssignStatement *>;
	using HorseIR::FlowAnalysisValue<Type>::Equals;

	static void Print(std::ostream& os, const Type *val)
	{
		os << "[";
		bool first = true;
		for (const auto& statement : *val)
		{
			if (!first)
			{
				os << ", ";
			}
			first = false;
			os << HorseIR::PrettyPrinter::PrettyString(statement->GetExpression());
		}
		os << "]";
	}
};

using ReachingDefinitionsProperties = HorseIR::FlowAnalysisMap<SymbolObject, ReachingDefinitionsValue>; 

class ReachingDefinitions : public HorseIR::ForwardAnalysis<ReachingDefinitionsProperties>
{
public:
	using Properties = ReachingDefinitionsProperties;
	using HorseIR::ForwardAnalysis<Properties>::ForwardAnalysis;

	void Visit(const HorseIR::AssignStatement *assignS) override;
	void Visit(const HorseIR::BlockStatement *blockS) override;

	Properties InitialFlow() const override;
	Properties Merge(const Properties& s1, const Properties& s2) const override;
};

}
