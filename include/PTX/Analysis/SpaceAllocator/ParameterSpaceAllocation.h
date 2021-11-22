#pragma once

#include <string>

#include "SASS/Tree/Constants.h"

#include "Libraries/robin_hood.h"

namespace PTX {
namespace Analysis {

class ParameterSpaceAllocation
{
public:
	ParameterSpaceAllocation(std::size_t parameterOffset) : m_parameterOffset(parameterOffset) {}
	// Parameters

	void AddParameter(const std::string& name, std::size_t size);
	bool ContainsParameter(const std::string& name) const;

	std::size_t GetParameterOffset(const std::string& name) const;

	// Formatting

	std::string ToString() const;

private:
	robin_hood::unordered_map<std::string, std::size_t> m_parameterMap;
	std::size_t m_parameterOffset = 0;
};

}
}
