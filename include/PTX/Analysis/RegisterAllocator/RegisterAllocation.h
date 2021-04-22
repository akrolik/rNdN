#pragma once

#include <cstdint>
#include <string>
#include <utility>

#include "Libraries/robin_hood.h"

namespace PTX {
namespace Analysis {

class RegisterAllocation
{
public:
	constexpr static unsigned int MaxRegister = 255;
	constexpr static unsigned int MaxPredicate = 7;

	// Registers

	void AddRegister(const std::string& name, std::uint8_t reg, std::uint8_t range = 1);
	bool ContainsRegister(const std::string& name) const;
	const std::pair<std::uint8_t, std::uint8_t>& GetRegister(const std::string& name) const;

	// Predicate

	void AddPredicate(const std::string& name, std::uint8_t reg);
	bool ContainsPredicate(const std::string& name) const;
	std::uint8_t GetPredicate(const std::string& name) const;

	// Counts

	std::uint8_t GetRegisterCount() const { return m_registerCount; }
	std::uint8_t GetPredicateCount() const { return m_predicateCount; }

	// Formatting

	std::string ToString() const;

private:
	robin_hood::unordered_map<std::string, std::pair<std::uint8_t, std::uint8_t>> m_registerMap;
	robin_hood::unordered_map<std::string, std::uint8_t> m_predicateMap;

	std::uint8_t m_registerCount = 0;
	std::uint8_t m_predicateCount = 0;
};

}
}
