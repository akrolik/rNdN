#include "Analysis/BasicFlow/LiveVariables.h"

namespace Analysis {

void LiveVariables::Kill(const HorseIR::SymbolTable::Symbol *symbol)
{
	// Remove all matches in the set
	
	m_currentInSet.erase(symbol);
}

void LiveVariables::Visit(const HorseIR::VariableDeclaration *declaration)
{
	// Kill all declarations

	m_currentInSet = m_currentOutSet;
	Kill(declaration->GetSymbol());
}

void LiveVariables::Visit(const HorseIR::AssignStatement *assignS)
{
	// Traverse the targets first to kill, then uses to add

	m_isTarget = true;

	for (const auto target : assignS->GetTargets())
	{
		target->Accept(*this);
		PropagateNext();
	}

	m_isTarget = false;

	assignS->GetExpression()->Accept(*this);
}

void LiveVariables::Visit(const HorseIR::Identifier *identifier)
{
	// Kill all definitions, add all uses

	m_currentInSet = m_currentOutSet;
	if (m_isTarget)
	{
		Kill(identifier->GetSymbol());
	}
	else
	{
		m_currentInSet.insert(identifier->GetSymbol());
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

	Properties outSet;

	outSet.insert(s1.begin(), s1.end());
	outSet.insert(s2.begin(), s2.end());

	return outSet;
}

}
