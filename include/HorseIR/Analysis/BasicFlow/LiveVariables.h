#pragma once

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

	inline const static std::string Name = "Live variables";
	inline const static std::string ShortName = "live";
	
	LiveVariables(const Program *program) : BackwardAnalysis<LiveVariablesProperties>(Name, ShortName, program) {}

	// Visitors

	void Visit(const VariableDeclaration *declaration) override;
	void Visit(const AssignStatement *assignS) override;
	void Visit(const Identifier *identifier) override;

	// Flow

	Properties InitialFlow() const override;
	Properties Merge(const Properties& s1, const Properties& s2) const override;

protected:
	void Kill(const SymbolTable::Symbol *symbol);

	bool m_isTarget = false;
};

}
}
