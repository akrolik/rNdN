#include "PTX/Analysis/SpaceAllocator/ParameterSpaceAllocation.h"

namespace PTX {
namespace Analysis {

// Parameters

void ParameterSpaceAllocation::AddParameter(const std::string& name, std::size_t size)
{
	m_parameterMap[name] = m_parameterOffset;
	m_parameterOffset += size;
}

bool ParameterSpaceAllocation::ContainsParameter(const std::string& name) const
{
	return m_parameterMap.find(name) != m_parameterMap.end();
}

std::size_t ParameterSpaceAllocation::GetParameterOffset(const std::string& name) const
{
	return m_parameterMap.at(name);
}

// Formatting

std::string ParameterSpaceAllocation::ToString() const
{
	std::string string = "  - Parameters = " + std::to_string(m_parameterOffset) + " bytes";
	for (const auto& [name, offset] : m_parameterMap)
	{
		string += "\n    - " + name + "-> { offset = " + Utils::Format::HexString(offset) + " bytes }";
	}
	return string;
}

}
}
