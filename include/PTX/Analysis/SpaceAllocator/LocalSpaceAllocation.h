#pragma once

#include <string>
#include <unordered_map>

#include "SASS/SASS.h"

namespace PTX {
namespace Analysis {

class LocalSpaceAllocation
{
public:
	LocalSpaceAllocation(std::size_t sharedMemoryOffset) : m_sharedMemoryOffset(sharedMemoryOffset) {}

	// Parameters

	void AddParameter(const std::string& name, std::size_t size);
	bool ContainsParameter(const std::string& name) const;

	std::size_t GetParameterOffset(const std::string& name) const;

	// Shared

	void AddSharedMemory(const std::string& name, std::size_t size);
	bool ContainsSharedMemory(const std::string& name) const;

	std::size_t GetSharedMemoryOffset(const std::string& name) const;
	std::size_t GetSharedMemorySize() const { return m_sharedMemorySize; }
	std::size_t GetDynamicSharedMemoryOffset() const { return m_sharedMemoryOffset + m_sharedMemorySize; }

	// Formatting

	std::string ToString() const;

private:
	std::unordered_map<std::string, std::size_t> m_parameterMap;
	std::size_t m_parameterOffset = SASS::CBANK_ParametersOffset;

	std::unordered_map<std::string, std::size_t> m_sharedMemoryMap;
	std::size_t m_sharedMemorySize = 0;
	std::size_t m_sharedMemoryOffset = 0;
};

}
}
