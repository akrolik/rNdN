#include "Analysis/ReachingDefinitions.h"

namespace Analysis {

void ReachingDefinitions::Visit(const HorseIR::AssignStatement *assignS)
{
	// For each target of the assignment, kill the previous definition (if any)
	// and add a new value to the set associating the target to the assignment

	ForwardAnalysis<ReachingDefinitionsValue>::Visit(assignS);

	for (const auto target : assignS->GetTargets())
	{
		// Construct the new value for the set

		auto symbol = target->GetSymbol();
		auto value = new ReachingDefinitionsValue(symbol, assignS);

		// Find the old value for the target, remove if found

		auto it = m_currentOutSet.begin();
		while (it != m_currentOutSet.end())
		{
			auto outVal = *it;
			if (target->GetSymbol() == outVal->GetSymbol())
			{
				m_currentOutSet.erase(it);
				break;
			}
			++it;
		}
		m_currentOutSet.insert(value);
	}
}

void ReachingDefinitions::Visit(const HorseIR::BlockStatement *blockS)
{
	// Visit all statements within the block and compute the sets

	ForwardAnalysis<ReachingDefinitionsValue>::Visit(blockS);

	// Kill all declarations that were part of the block

	auto symbolTable = blockS->GetSymbolTable();
	auto it = m_currentOutSet.begin();
	while (it != m_currentOutSet.end())
	{
		auto val = *it;
		if (symbolTable->ContainsSymbol(val->GetSymbol()))
		{
			it = m_currentOutSet.erase(it);
		}
		else
		{
			++it;
		}
	}
}

ReachingDefinitions::SetType ReachingDefinitions::Merge(const SetType& s1, const SetType& s2) const
{
	// Merge the sets using a union on each target set. Avoid duplicate targets

	ReachingDefinitions::SetType outSet(s1);
	for (const auto val : s2)
	{
		bool match = false;

		auto it = outSet.begin();
		while (it != outSet.end())
		{
			const auto outVal = *it;
			if (val->GetSymbol() == outVal->GetSymbol())
			{
				auto newVal = new ReachingDefinitionsValue(val->GetSymbol(), val->GetStatements());
				newVal->AddStatements(outVal->GetStatements());
				outSet.erase(it);
				outSet.insert(newVal);

				match = true;
				break;
			}
			++it;
		}

		if (!match)
		{
			outSet.insert(val);
		}
	}
	return outSet;
}

}
