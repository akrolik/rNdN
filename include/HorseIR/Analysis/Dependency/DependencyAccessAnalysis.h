#pragma once

#include <sstream>
#include <unordered_set>

#include "Analysis/FlowValue.h"

#include "HorseIR/Analysis/Framework/ForwardAnalysis.h"

#include "HorseIR/Analysis/Utils/SymbolObject.h"
#include "HorseIR/Tree/Tree.h"
#include "HorseIR/Utils/PrettyPrinter.h"

namespace HorseIR {
namespace Analysis {

struct DependencyAccessValue : ::Analysis::Value<std::unordered_set<const Statement *>>
{
	using Type = std::unordered_set<const Statement *>;
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
			os << PrettyPrinter::PrettyString(statement, true);
		}
		os << "]";
	}
};

using DependencyAccessProperties = ::Analysis::Pair<
	::Analysis::Map<SymbolObject, DependencyAccessValue>,
	::Analysis::Map<SymbolObject, DependencyAccessValue>
>;

class DependencyAccessAnalysis : public ForwardAnalysis<DependencyAccessProperties>
{
public:
	using Properties = DependencyAccessProperties;
	using ForwardAnalysis<Properties>::ForwardAnalysis;

	void Visit(const DeclarationStatement *declarationS) override;
	void Visit(const AssignStatement *assignS) override;
	void Visit(const Identifier *identifier) override;

	Properties InitialFlow() const override;
	Properties Merge(const Properties& s1, const Properties& s2) const override;

	std::string Name() const override { return "Dependency access analysis"; }
};

}
}
