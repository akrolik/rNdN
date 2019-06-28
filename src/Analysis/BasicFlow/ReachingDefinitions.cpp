#include "Analysis/BasicFlow/ReachingDefinitions.h"

namespace Analysis {

void ReachingDefinitions::Visit(const HorseIR::AssignStatement *assignS)
{
	// For each target of the assignment, kill the previous definition (if any)
	// and add a new value to the map associating the target to the assignment

	m_currentOutSet = m_currentInSet;

	for (const auto target : assignS->GetTargets())
	{
		// Construct the new value for the set

		auto symbol = target->GetSymbol();
		m_currentOutSet[symbol] = new ReachingDefinitionsValue::Type({assignS});
	}
}

void ReachingDefinitions::Visit(const HorseIR::BlockStatement *blockS)
{
	// Visit all statements within the block and compute the sets

	ForwardAnalysis<ReachingDefinitionsProperties>::Visit(blockS);

	// Kill all declarations that were part of the block

	auto symbolTable = blockS->GetSymbolTable();
	auto it = m_currentOutSet.begin();
	while (it != m_currentOutSet.end())
	{
		auto symbol = it->first;
		if (symbolTable->ContainsSymbol(symbol))
		{
			it = m_currentOutSet.erase(it);
		}
		else
		{
			++it;
		}
	}
}

ReachingDefinitions::Properties ReachingDefinitions::InitialFlow() const
{
	// Initial flow is empty set, no definitions reach!

	Properties initialFlow;
	return initialFlow;
}

ReachingDefinitions::Properties ReachingDefinitions::Merge(const Properties& s1, const Properties& s2) const
{
	// Merge the maps using union. Find if the symbol is already present, and either merge the
	// statements or copy across the pointers

	Properties outSet(s1);
	for (const auto val : s2)
	{
		auto it = outSet.find(val.first);
		if (it != outSet.end())
		{
			auto newSet = new ReachingDefinitionsValue::Type();
			newSet->insert(val.second->begin(), val.second->end());
			newSet->insert(it->second->begin(), it->second->end());
			outSet[val.first] = newSet;
		}
		else
		{
			outSet.insert(val);
		}
	}
	return outSet;
}

}
