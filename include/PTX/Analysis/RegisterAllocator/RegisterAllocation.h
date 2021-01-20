#pragma once

#include <cstdint>
#include <string>
#include <unordered_map>
#include <utility>

namespace PTX {
namespace Analysis {

class RegisterAllocation
{
public:
	constexpr static unsigned int MaxRegister = 255;
	constexpr static unsigned int MaxPredicate = 7;

	// Registers

	void AddRegister(const std::string& name, std::uint8_t reg, std::uint8_t range = 1)
	{
		m_registerMap[name] = {reg, range};

		auto maxRegister = reg + (range - 1);
		if (maxRegister > m_registerCount)
		{
			m_registerCount = maxRegister;
		}
	}

	bool ContainsRegister(const std::string& name)
	{
		return m_registerMap.find(name) != m_registerMap.end();
	}

	const std::pair<std::uint8_t, std::uint8_t>& GetRegister(const std::string& name)
	{
		return m_registerMap.at(name);
	}

	// Predicate

	void AddPredicate(const std::string& name, std::uint8_t reg)
	{
		m_predicateMap[name] = reg;

		auto maxPredicate = reg + 1;
		if (maxPredicate > m_predicateCount)
		{
			m_predicateCount = maxPredicate;
		}
	}

	bool ContainsPredicate(const std::string& name)
	{
		return m_predicateMap.find(name) != m_predicateMap.end();
	}

	std::uint8_t GetPredicate(const std::string& name)
	{
		return m_predicateMap.at(name);
	}

	// Counts

	std::uint8_t GetRegisterCount() const
	{
		return m_registerCount;
	}

	std::uint8_t GetPredicateCount() const
	{
		return m_predicateCount;
	}

	// Formatting

	std::string ToString() const
	{
		std::string string = "  - Registers = " + std::to_string(m_registerCount) + " + [R0 dummy]";
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

private:
	std::unordered_map<std::string, std::pair<std::uint8_t, std::uint8_t>> m_registerMap;
	std::unordered_map<std::string, std::uint8_t> m_predicateMap;

	std::uint8_t m_registerCount = 0;
	std::uint8_t m_predicateCount = 0;
};

}
}
