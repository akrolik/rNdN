#pragma once

#include <string>
#include <unordered_map>

#include "SASS/SASS.h"

namespace PTX {
namespace Analysis {

class SpaceAllocation
{
public:
	// Parameters

	void AddParameter(const std::string& name, std::size_t size);
	bool ContainsParameter(const std::string& name) const;

	std::size_t GetParameterOffset(const std::string& name) const;

	// Shared

	void AddSharedVariable(const std::string& name, std::size_t size);
	bool ContainsSharedVariable(const std::string& name) const;

	std::size_t GetSharedVariableOffset(const std::string& name) const;
	std::size_t GetDynamicSharedVariableOffset() const;

	// Formatting

	std::string ToString() const;

private:
	std::unordered_map<std::string, std::size_t> m_parameterMap;
	std::size_t m_parameterOffset = SASS::CBANK_ParametersOffset;

	std::unordered_map<std::string, std::size_t> m_sharedMap;
	std::size_t m_sharedOffset = 0;
};

}
}
