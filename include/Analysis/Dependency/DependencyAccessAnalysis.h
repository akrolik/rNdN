#pragma once

#include <sstream>
#include <unordered_set>

#include "HorseIR/Analysis/ForwardAnalysis.h"

#include "Analysis/Utils/SymbolObject.h"

#include "HorseIR/Tree/Tree.h"
#include "HorseIR/Utils/PrettyPrinter.h"

namespace Analysis {

struct DependencyAccessValue : HorseIR::FlowAnalysisValue<std::unordered_set<const HorseIR::Statement *>>
{
	using Type = std::unordered_set<const HorseIR::Statement *>;
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
			os << HorseIR::PrettyPrinter::PrettyString(statement, true);
		}
		os << "]";
	}
};

using DependencyAccessProperties = HorseIR::FlowAnalysisPair<
	HorseIR::FlowAnalysisMap<SymbolObject, DependencyAccessValue>,
	HorseIR::FlowAnalysisMap<SymbolObject, DependencyAccessValue>
>;

class DependencyAccessAnalysis : public HorseIR::ForwardAnalysis<DependencyAccessProperties>
{
public:
	using Properties = DependencyAccessProperties;
	using HorseIR::ForwardAnalysis<Properties>::ForwardAnalysis;

	void Visit(const HorseIR::AssignStatement *assignS) override;
	void Visit(const HorseIR::Identifier *identifier) override;

	Properties InitialFlow() const override;
	Properties Merge(const Properties& s1, const Properties& s2) const override;

	std::string Name() const override { return "Dependency access analysis"; }
};

}
