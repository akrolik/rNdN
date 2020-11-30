#pragma once

#include <cstdint>
#include <string>
#include <unordered_map>

namespace PTX {
namespace Analysis {

class RegisterAllocation
{
public:
	// Registers

	void AddRegister(const std::string& name, std::uint8_t reg)
	{
		m_registerMap[name] = reg;
		if (reg + 1 > m_count)
		{
			m_count = reg + 1;
		}
	}

	bool ContainsRegister(const std::string& name)
	{
		return m_registerMap.find(name) != m_registerMap.end();
	}

	std::uint8_t GetRegister(const std::string& name)
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
		for (const auto& [name, reg] : m_registerMap)
		{
			string += "\n    - " + name + "->R" + std::to_string(reg);
		}
		string += "\n  - Predicates";
		for (const auto& [name, reg] : m_predicateMap)
		{
			string += "\n    - " + name + "->P" + std::to_string(reg);
		}
		return string;
	}

private:
	std::unordered_map<std::string, std::uint8_t> m_registerMap;
	std::unordered_map<std::string, std::uint8_t> m_predicateMap;

	std::uint8_t m_count = 0;
};

}
}
