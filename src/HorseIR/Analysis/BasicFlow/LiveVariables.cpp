#include "HorseIR/Analysis/BasicFlow/LiveVariables.h"

namespace HorseIR {
namespace Analysis {

void LiveVariables::Kill(const SymbolTable::Symbol *symbol)
{
	// Remove all matches in the set
	
	m_currentSet.erase(symbol);
}

void LiveVariables::Visit(const VariableDeclaration *declaration)
{
	// Kill all declarations

	Kill(declaration->GetSymbol());
}

void LiveVariables::Visit(const AssignStatement *assignS)
{
	// Traverse the targets first to kill, then uses to add

	m_isTarget = true;

	for (const auto target : assignS->GetTargets())
	{
		target->Accept(*this);
	}

	m_isTarget = false;

	assignS->GetExpression()->Accept(*this);
}

void LiveVariables::Visit(const Identifier *identifier)
{
	// Kill all definitions, add all uses

	if (m_isTarget)
	{
		Kill(identifier->GetSymbol());
	}
	else
	{
		m_currentSet.insert(identifier->GetSymbol());
	}
}

LiveVariables::Properties LiveVariables::InitialFlow() const
{
	// Initial flow is the empty set, no variables are live!

        Properties initialFlow;
	return initialFlow;
}

LiveVariables::Properties LiveVariables::Merge(const Properties& s1, const Properties& s2) const
{
	// Simple merge operation, add all non-duplicate eelements to the out set

	Properties outSet(s1);

	outSet.insert(s2.begin(), s2.end());

	return outSet;
}

}
}
