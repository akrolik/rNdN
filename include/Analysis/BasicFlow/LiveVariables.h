#pragma once

#include <sstream>

#include "HorseIR/Analysis/BackwardAnalysis.h"

#include "Analysis/Utils/SymbolObject.h"

#include "HorseIR/Tree/Tree.h"

namespace Analysis {

using LiveVariablesProperties = HorseIR::FlowAnalysisSet<SymbolObject>;

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

	std::string Name() const override { return "Live variables"; }

protected:
	void Kill(const HorseIR::SymbolTable::Symbol *symbol);

	bool m_isTarget = false;
};

}
