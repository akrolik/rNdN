#include "PTX/Analysis/RegisterAllocator/RegisterAllocation.h"

namespace PTX {
namespace Analysis {

// Registers

void RegisterAllocation::AddRegister(const std::string& name, std::uint8_t reg, std::uint8_t range)
{
	m_registerMap.emplace(name, std::make_pair(reg, range));

	auto maxRegister = reg + range;
	if (maxRegister > m_registerCount)
	{
		m_registerCount = maxRegister;
	}
}

bool RegisterAllocation::ContainsRegister(const std::string& name) const
{
	return m_registerMap.find(name) != m_registerMap.end();
}

const std::pair<std::uint8_t, std::uint8_t>& RegisterAllocation::GetRegister(const std::string& name) const
{
	return m_registerMap.at(name);
}

// Predicate

void RegisterAllocation::AddPredicate(const std::string& name, std::uint8_t reg)
{
	m_predicateMap.emplace(name, reg);

	auto maxPredicate = reg + 1;
	if (maxPredicate > m_predicateCount)
	{
		m_predicateCount = maxPredicate;
	}
}

bool RegisterAllocation::ContainsPredicate(const std::string& name) const
{
	return m_predicateMap.find(name) != m_predicateMap.end();
}

std::uint8_t RegisterAllocation::GetPredicate(const std::string& name) const
{
	return m_predicateMap.at(name);
}

// Formatting

std::string RegisterAllocation::ToString() const
{
	std::string string = "  - Registers = " + std::to_string(m_registerCount);
	for (const auto& [name, pair] : m_registerMap)
	{
		string += "\n    - " + name + "->";

		auto reg = pair.first;
		auto range = pair.second;

		if (range > 1)
		{
			string += "{";
		}

		auto first = true;
		for (auto i = 0u; i < range; ++i)
		{
			if (!first)
			{
				string += ", ";
			}
			first = false;
			string += "R" + std::to_string(reg + i);
		}

		if (range > 1)
		{
			string += "}";
		}
	}

	string += "\n  - Predicates = " + std::to_string(m_predicateCount);
	for (const auto& [name, reg] : m_predicateMap)
	{
		string += "\n    - " + name + "->P" + std::to_string(reg);
	}
	return string;
}

}
}
