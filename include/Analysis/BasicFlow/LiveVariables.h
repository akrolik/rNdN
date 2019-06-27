#pragma once

#include <sstream>

#include "HorseIR/Analysis/BackwardAnalysis.h"

#include "HorseIR/Semantics/SymbolTable/SymbolTable.h"
#include "HorseIR/Tree/Tree.h"
#include "HorseIR/Utils/PrettyPrinter.h"

namespace Analysis {

struct LiveVariablesValue : HorseIR::FlowAnalysisPointerValue<HorseIR::SymbolTable::Symbol>
{
	using Type = HorseIR::SymbolTable::Symbol;
	using HorseIR::FlowAnalysisPointerValue<Type>::Equals;

	struct Hash
	{
		bool operator()(const Type *val) const
		{
			return std::hash<const Type *>()(val);
		}
	};

	static void Print(std::ostream& os, const Type *val)
	{
		os << HorseIR::PrettyPrinter::PrettyString(val->node);
	}
};

using LiveVariablesProperties = HorseIR::FlowAnalysisSet<LiveVariablesValue>;

class LiveVariables : public HorseIR::BackwardAnalysis<LiveVariablesProperties>
{
public:
	using Properties = LiveVariablesProperties;
	using HorseIR::BackwardAnalysis<LiveVariablesProperties>::BackwardAnalysis;

	void Visit(const HorseIR::VariableDeclaration *declaration) override;
	void Visit(const HorseIR::AssignStatement *assignS) override;
	void Visit(const HorseIR::Identifier *identifier) override;

	Properties InitialFlow() const override;
	Properties Merge(const Properties& s1, const Properties& s2) const override;

protected:
	void Kill(const HorseIR::SymbolTable::Symbol *symbol);

	bool m_isTarget = false;
};

}
