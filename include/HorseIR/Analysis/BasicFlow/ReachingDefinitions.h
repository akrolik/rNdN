#pragma once

#include <sstream>
#include <unordered_set>

#include "HorseIR/Analysis/Framework/ForwardAnalysis.h"
#include "HorseIR/Analysis/Utils/SymbolObject.h"

#include "HorseIR/Tree/Tree.h"
#include "HorseIR/Utils/PrettyPrinter.h"

namespace HorseIR {
namespace Analysis {

struct ReachingDefinitionsValue : FlowAnalysisValue<std::unordered_set<const AssignStatement *>>
{
	using Type = std::unordered_set<const AssignStatement *>;
	using FlowAnalysisValue<Type>::Equals;

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
			os << PrettyPrinter::PrettyString(statement->GetExpression());
		}
		os << "]";
	}
};

using ReachingDefinitionsProperties = FlowAnalysisMap<SymbolObject, ReachingDefinitionsValue>; 

class ReachingDefinitions : public ForwardAnalysis<ReachingDefinitionsProperties>
{
public:
	using Properties = ReachingDefinitionsProperties;
	using ForwardAnalysis<Properties>::ForwardAnalysis;

	void Visit(const AssignStatement *assignS) override;
	void Visit(const BlockStatement *blockS) override;

	Properties InitialFlow() const override;
	Properties Merge(const Properties& s1, const Properties& s2) const override;

	std::string Name() const override { return "Reaching definitions"; }
};

}
}
