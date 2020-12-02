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
	// Registers

	void AddRegister(const std::string& name, std::uint8_t reg, std::uint8_t range = 1)
	{
		m_registerMap[name] = {reg, range};

		auto maxReg = reg + range + 1;
		if (maxReg > m_count)
		{
			m_count = maxReg;
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

	std::uint8_t GetCount() const
	{
		return m_count;
	}

	// Formatting

	std::string ToString() const
	{
		std::string string = "  - Registers = " + std::to_string(m_count);
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
			for (auto i = 0; i < range; ++i)
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

		string += "\n  - Predicates";
		for (const auto& [name, reg] : m_predicateMap)
		{
			string += "\n    - " + name + "->P" + std::to_string(reg);
		}
		return string;
	}

private:
	std::unordered_map<std::string, std::pair<std::uint8_t, std::uint8_t>> m_registerMap;
	std::unordered_map<std::string, std::uint8_t> m_predicateMap;

	std::uint8_t m_count = 0;
};

}
}
