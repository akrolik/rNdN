#pragma once

#include <sstream>

#include "Analysis/FlowValue.h"

#include "HorseIR/Analysis/Framework/ForwardAnalysis.h"
#include "HorseIR/Analysis/Utils/SymbolObject.h"

#include "HorseIR/Tree/Tree.h"
#include "HorseIR/Utils/PrettyPrinter.h"

namespace HorseIR {
namespace Analysis {

struct ReachingDefinitionsValue : ::Analysis::Value<robin_hood::unordered_set<const AssignStatement *>>
{
	using Type = robin_hood::unordered_set<const AssignStatement *>;
	using ::Analysis::Value<Type>::Equals;

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

using ReachingDefinitionsProperties = ::Analysis::Map<SymbolObject, ReachingDefinitionsValue>; 

class ReachingDefinitions : public ForwardAnalysis<ReachingDefinitionsProperties>
{
public:
	using Properties = ReachingDefinitionsProperties;

	inline const static std::string Name = "Reaching definitions";
	inline const static std::string ShortName = "rdef";
	
	ReachingDefinitions(const Program *program) : ForwardAnalysis<ReachingDefinitionsProperties>(Name, ShortName, program) {}

	// Visitors

	void Visit(const AssignStatement *assignS) override;
	void Visit(const BlockStatement *blockS) override;

	// Flow

	Properties InitialFlow() const override;
	Properties Merge(const Properties& s1, const Properties& s2) const override;
};

}
}
