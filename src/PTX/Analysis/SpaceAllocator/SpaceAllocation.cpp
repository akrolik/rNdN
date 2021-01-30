#include "PTX/Analysis/SpaceAllocator/SpaceAllocation.h"

namespace PTX {
namespace Analysis {

// Parameters

void SpaceAllocation::AddParameter(const std::string& name, std::size_t size)
{
	m_parameterMap[name] = m_parameterOffset;
	m_parameterOffset += size;
}

bool SpaceAllocation::ContainsParameter(const std::string& name) const
{
	return m_parameterMap.find(name) != m_parameterMap.end();
}

std::size_t SpaceAllocation::GetParameterOffset(const std::string& name) const
{
	return m_parameterMap.at(name);
}

// Predicate

void SpaceAllocation::AddSharedVariable(const std::string& name, std::size_t size)
{
	m_sharedMap[name] = m_sharedOffset;
	m_sharedOffset +=  size;
}

bool SpaceAllocation::ContainsSharedVariable(const std::string& name) const
{
	return m_sharedMap.find(name) != m_sharedMap.end();
}

std::size_t SpaceAllocation::GetSharedVariableOffset(const std::string& name) const
{
	return m_sharedMap.at(name);
}

// Formatting

std::string SpaceAllocation::ToString() const
{
	std::string string = "  - Parameters = " + std::to_string(m_parameterOffset);
	for (const auto& [name, offset] : m_parameterMap)
	{
		string += "\n    - " + name + "->" + Utils::Format::HexString(offset);
	}

	string += "\n  - Shared Variables = " + std::to_string(m_sharedOffset);
	for (const auto& [name, offset] : m_sharedMap)
	{
		string += "\n    - " + name + "->" + Utils::Format::HexString(offset);
	}
	return string;
}

}
}
