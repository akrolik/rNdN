#pragma once

#include <sstream>

#include "Analysis/FlowValue.h"

#include "HorseIR/Analysis/Framework/BackwardAnalysis.h"
#include "HorseIR/Analysis/Utils/SymbolObject.h"

#include "HorseIR/Tree/Tree.h"

namespace HorseIR {
namespace Analysis {

using LiveVariablesProperties = ::Analysis::Set<SymbolObject>;

class LiveVariables : public BackwardAnalysis<LiveVariablesProperties>
{
public:
	using Properties = LiveVariablesProperties;
	using BackwardAnalysis<LiveVariablesProperties>::BackwardAnalysis;

	void Visit(const VariableDeclaration *declaration) override;
	void Visit(const AssignStatement *assignS) override;
	void Visit(const Identifier *identifier) override;

	Properties InitialFlow() const override;
	Properties Merge(const Properties& s1, const Properties& s2) const override;

	std::string Name() const override { return "Live variables"; }

protected:
	void Kill(const SymbolTable::Symbol *symbol);

	bool m_isTarget = false;
};

}
}
