#include "HorseIR/Analysis/BasicFlow/ReachingDefinitions.h"

namespace HorseIR {
namespace Analysis {

void ReachingDefinitions::Visit(const AssignStatement *assignS)
{
	// For each target of the assignment, kill the previous definition (if any)
	// and add a new value to the map associating the target to the assignment

	for (const auto target : assignS->GetTargets())
	{
		// Construct the new value for the set

		auto symbol = target->GetSymbol();
		m_currentSet[symbol] = new ReachingDefinitionsValue::Type({assignS});
	}
}

void ReachingDefinitions::Visit(const BlockStatement *blockS)
{
	// Visit all statements within the block and compute the sets

	ForwardAnalysis<ReachingDefinitionsProperties>::Visit(blockS);

	// Kill all declarations that were part of the block

	auto symbolTable = blockS->GetSymbolTable();
	auto it = m_currentSet.begin();
	while (it != m_currentSet.end())
	{
		auto symbol = it->first;
		if (symbolTable->ContainsSymbol(symbol))
		{
			it = m_currentSet.erase(it);
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
	for (const auto& [symbol, definitions] : s2)
	{
		auto it = outSet.find(symbol);
		if (it != outSet.end())
		{
			auto newSet = new ReachingDefinitionsValue::Type();
			newSet->insert(definitions->begin(), definitions->end());
			newSet->insert(it->second->begin(), it->second->end());
			outSet[symbol] = newSet;
		}
		else
		{
			outSet.insert({symbol, definitions});
		}
	}
	return outSet;
}

}
}
