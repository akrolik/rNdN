#include "Analysis/LiveVariables.h"

namespace Analysis {

void LiveVariables::Kill(const HorseIR::SymbolTable::Symbol *symbol)
{
	// Remove all matches in the set, note that we must be careful of the iterator

	auto it = m_currentInSet.begin();
	while(it != m_currentInSet.end())
	{
		auto value = *it;
		if (value->GetSymbol() == symbol)
		{
			it = m_currentInSet.erase(it);
		}
		else
		{
			++it;
		}
	}
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
	m_currentInSet = m_currentOutSet;

	// Kill all definitions, add all uses

	if (m_isTarget)
	{
		Kill(identifier->GetSymbol());
	}
	else
	{
		m_currentInSet.insert(new LiveVariablesValue(identifier->GetSymbol()));
	}
}

LiveVariables::SetType LiveVariables::Merge(const SetType& s1, const SetType& s2) const
{
	// Simple merge operation, add all non-duplicate eelements to the out set

	LiveVariables::SetType outSet;

	outSet.insert(s1.begin(), s1.end());
	outSet.insert(s2.begin(), s2.end());

	return outSet;
}

}
