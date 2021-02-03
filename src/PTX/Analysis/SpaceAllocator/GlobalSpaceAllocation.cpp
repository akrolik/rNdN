#include "PTX/Analysis/SpaceAllocator/GlobalSpaceAllocation.h"

namespace PTX {
namespace Analysis {

// Global Memory

void GlobalSpaceAllocation::AddGlobalMemory(const std::string& name, std::size_t size)
{
	m_globalMemoryMap[name] = { m_globalMemorySize, size };
	m_globalMemorySize +=  size;
}

bool GlobalSpaceAllocation::ContainsGlobalMemory(const std::string& name) const
{
	return m_globalMemoryMap.find(name) != m_globalMemoryMap.end();
}

std::size_t GlobalSpaceAllocation::GetGlobalMemoryOffset(const std::string& name) const
{
	return m_globalMemoryMap.at(name).first;
}

std::size_t GlobalSpaceAllocation::GetGlobalMemorySize(const std::string& name) const
{
	return m_globalMemoryMap.at(name).second;
}

// Shared Memory

void GlobalSpaceAllocation::AddSharedMemory(const std::string& name, std::size_t size)
{
	m_sharedMemoryMap[name] = m_sharedMemorySize;
	m_sharedMemorySize +=  size;
}

bool GlobalSpaceAllocation::ContainsSharedMemory(const std::string& name) const
{
	return m_sharedMemoryMap.find(name) != m_sharedMemoryMap.end();
}

std::size_t GlobalSpaceAllocation::GetSharedMemoryOffset(const std::string& name) const
{
	return m_sharedMemoryMap.at(name);
}

void GlobalSpaceAllocation::AddDynamicSharedMemory(const std::string& name)
{
	m_dynamicSharedMemorySet.insert(name);
}

bool GlobalSpaceAllocation::ContainsDynamicSharedMemory(const std::string& name) const
{
	return m_dynamicSharedMemorySet.find(name) != m_dynamicSharedMemorySet.end();
}

// Formatting

std::string GlobalSpaceAllocation::ToString() const
{
	std::string string = "  - Global Memory";
	for (const auto& [name, info] : m_globalMemoryMap)
	{
		string += "\n    - " + name + "->{" + Utils::Format::HexString(info.first) + "," + Utils::Format::HexString(info.second) + "}";
	}
	string += "\n  - Shared Memory = " + std::to_string(m_sharedMemorySize) + " bytes";
	for (const auto& [name, offset] : m_sharedMemoryMap)
	{
		string += "\n    - " + name + "->" + Utils::Format::HexString(offset);
	}
	string += "\n  - Dynamic Shared Memory";
	for (const auto& name : m_dynamicSharedMemorySet)
	{
		string += "\n    - " + name;
	}
	return string;
}

}
}
