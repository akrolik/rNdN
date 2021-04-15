#pragma once

#include <string>
#include <unordered_map>

#include "SASS/Tree/Constants.h"

namespace PTX {
namespace Analysis {

class ParameterSpaceAllocation
{
public:
	// Parameters

	void AddParameter(const std::string& name, std::size_t size);
	bool ContainsParameter(const std::string& name) const;

	std::size_t GetParameterOffset(const std::string& name) const;

	// Formatting

	std::string ToString() const;

private:
	std::unordered_map<std::string, std::size_t> m_parameterMap;
	std::size_t m_parameterOffset = SASS::CBANK_ParametersOffset;
};

}
}
