#include "Analysis/Dependency/DependencyAccessAnalysis.h"

namespace Analysis {

void DependencyAccessAnalysis::Visit(const HorseIR::DeclarationStatement *declarationS)
{
	m_currentOutSet = m_currentInSet;

	// Add an implicit write for declarations

	auto symbol = declarationS->GetDeclaration()->GetSymbol();
	m_currentOutSet.second[symbol] = new DependencyAccessValue::Type({declarationS});
}

void DependencyAccessAnalysis::Visit(const HorseIR::AssignStatement *assignS)
{
	m_currentOutSet = m_currentInSet;

	// Traverse the reads first, since they may be later killed by the writes

	assignS->GetExpression()->Accept(*this);

	// For each write, kill the previous reads and writes, then add the write

	for (const auto& target : assignS->GetTargets())
	{
		auto symbol = target->GetSymbol();

		m_currentOutSet.first.erase(symbol);
		m_currentOutSet.second[symbol] = new DependencyAccessValue::Type({assignS});
	}
}

void DependencyAccessAnalysis::Visit(const HorseIR::Identifier *identifier)
{
	// Get the set of reads and the symbol

	auto& reads = m_currentOutSet.first;
	auto symbol = identifier->GetSymbol();

	// Add the statement to the input set of statements if they exist

	auto newSet = new DependencyAccessValue::Type({m_currentStatement});

	const auto& it = reads.find(symbol);
	if (it != reads.end())
	{
		newSet->insert(it->second->begin(), it->second->end());
	}
	reads[symbol] = newSet;
}

DependencyAccessAnalysis::Properties DependencyAccessAnalysis::InitialFlow() const
{
	// Initial set of properties is empty

	Properties properties;
	return properties;
}

DependencyAccessAnalysis::Properties DependencyAccessAnalysis::Merge(const Properties& s1, const Properties& s2) const
{
	// Merge the maps using union. Find if the symbol is already present, and either merge the
	// statements or copy across the pointers

	Properties outPair(s1);

	// Merge the reads from the second properties map

	for (const auto& [symbol, accesses] : s2.first)
	{
		// Convenience var for the output set of reads

		auto& reads = outPair.first;

		// Merge the reads or insert the second set

		auto it = reads.find(symbol);
		if (it != reads.end())
		{
			auto newSet = new DependencyAccessValue::Type();
			newSet->insert(accesses->begin(), accesses->end());
			newSet->insert(it->second->begin(), it->second->end());
			reads[symbol] = newSet;
		}
		else
		{
			reads.insert({symbol, accesses});
		}
	}

	// Merge the writes from the second properties map

	for (const auto& [symbol, accesses] : s2.second)
	{
		// Convenience var for the output set of writes

		auto& writes = outPair.second;

		// Merge the writes or insert the second set

		auto it = writes.find(symbol);
		if (it != writes.end())
		{
			auto newSet = new DependencyAccessValue::Type();
			newSet->insert(accesses->begin(), accesses->end());
			newSet->insert(it->second->begin(), it->second->end());
			writes[symbol] = newSet;
		}
		else
		{
			writes.insert({symbol, accesses});
		}
	}

	return outPair;
}

}
