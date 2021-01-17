#include "PTX/Analysis/BasicFlow/ReachingDefinitions.h"

namespace PTX {
namespace Analysis {

void ReachingDefinitions::Visit(const InstructionStatement *statement)
{
	// Copy input to output directly

	m_currentOutSet = m_currentInSet;
	m_currentStatement = statement;

	// Assume the first operand is the destination

	const auto operands = statement->GetOperands();
	if (operands.size() > 0)
	{
		const auto& destination = operands.at(0);
		destination->Accept(static_cast<ConstOperandDispatcher&>(*this));
	}

	m_currentStatement = nullptr;
}

template<class T>
void ReachingDefinitions::Visit(const Register<T> *reg)
{
	// Add the instruction statement to the map

	auto key = new ReachingDefinitionsKey::Type(reg->GetName());
	auto value = new ReachingDefinitionsValue::Type({m_currentStatement});
	m_currentOutSet[key] = value;
}

ReachingDefinitions::Properties ReachingDefinitions::InitialFlow() const
{
	return {};
}

ReachingDefinitions::Properties ReachingDefinitions::TemporaryFlow() const
{
	return {};
}

ReachingDefinitions::Properties ReachingDefinitions::Merge(const Properties& s1, const Properties& s2) const
{
	// Merge the map using union. Find the existing element with the name if present, and merge.

	Properties outSet(s1);
	for (const auto& [name, definitions] : s2)
	{
		auto it = outSet.find(name);
		if (it != outSet.end())
		{
			auto newSet = new ReachingDefinitionsValue::Type();
			newSet->insert(definitions->begin(), definitions->end());
			newSet->insert(it->second->begin(), it->second->end());
			outSet[name] = newSet;
		}
		else
		{
			outSet.insert({name, definitions});
		}
	}
	return outSet;
}

}
}
