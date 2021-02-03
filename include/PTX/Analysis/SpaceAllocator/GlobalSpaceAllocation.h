#pragma once

#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>

#include "SASS/SASS.h"

namespace PTX {
namespace Analysis {

class GlobalSpaceAllocation
{
public:
	// Global Memory

	void AddGlobalMemory(const std::string& name, std::size_t size);
	bool ContainsGlobalMemory(const std::string& name) const;

	std::size_t GetGlobalMemoryOffset(const std::string& name) const;
	std::size_t GetGlobalMemorySize(const std::string& name) const;
	std::size_t GetGlobalMemorySize() const { return m_globalMemorySize; }

	// Shared Memory

	void AddSharedMemory(const std::string& name, std::size_t size);
	bool ContainsSharedMemory(const std::string& name) const;

	std::size_t GetSharedMemoryOffset(const std::string& name) const;
	std::size_t GetSharedMemorySize() const { return m_sharedMemorySize; }

	void AddDynamicSharedMemory(const std::string& name);
	bool ContainsDynamicSharedMemory(const std::string& name) const;

	bool GetDynamicSharedMemory() const { return (m_dynamicSharedMemorySet.size() > 0); }

	// Formatting

	std::string ToString() const;

private:
	std::unordered_map<std::string, std::pair<std::size_t, std::size_t>> m_globalMemoryMap; // { offset, size }
	std::size_t m_globalMemorySize = 0;

	std::unordered_set<std::string> m_dynamicSharedMemorySet;
	std::unordered_map<std::string, std::size_t> m_sharedMemoryMap;
	std::size_t m_sharedMemorySize = 0;
};

}
}
