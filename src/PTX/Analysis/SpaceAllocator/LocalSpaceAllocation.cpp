#include "PTX/Analysis/SpaceAllocator/LocalSpaceAllocation.h"

namespace PTX {
namespace Analysis {

// Parameters

void LocalSpaceAllocation::AddParameter(const std::string& name, std::size_t size)
{
	m_parameterMap[name] = m_parameterOffset;
	m_parameterOffset += size;
}

bool LocalSpaceAllocation::ContainsParameter(const std::string& name) const
{
	return m_parameterMap.find(name) != m_parameterMap.end();
}

std::size_t LocalSpaceAllocation::GetParameterOffset(const std::string& name) const
{
	return m_parameterMap.at(name);
}

// Shared Memory

void LocalSpaceAllocation::AddSharedMemory(const std::string& name, std::size_t size)
{
	m_sharedMemoryMap[name] = m_sharedMemoryOffset + m_sharedMemorySize;
	m_sharedMemorySize +=  size;
}

bool LocalSpaceAllocation::ContainsSharedMemory(const std::string& name) const
{
	return m_sharedMemoryMap.find(name) != m_sharedMemoryMap.end();
}

std::size_t LocalSpaceAllocation::GetSharedMemoryOffset(const std::string& name) const
{
	return m_sharedMemoryMap.at(name);
}

// Formatting

std::string LocalSpaceAllocation::ToString() const
{
	std::string string = "  - Parameters = " + std::to_string(m_parameterOffset) + " bytes";
	for (const auto& [name, offset] : m_parameterMap)
	{
		string += "\n    - " + name + "->" + Utils::Format::HexString(offset);
	}

	string += "\n  - Shared Memory = " + std::to_string(m_sharedMemorySize) + " bytes (offset = " + std::to_string(m_sharedMemoryOffset) + " bytes)";
	for (const auto& [name, offset] : m_sharedMemoryMap)
	{
		string += "\n    - " + name + "->" + Utils::Format::HexString(offset);
	}
	return string;
}

}
}
