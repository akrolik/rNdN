#pragma once

#include <string>

#include "Libraries/json.hpp"

#include "Utils/Logger.h"

namespace Codegen { class InputOptions; }

namespace PTX {

class Function;
class FunctionOptions
{
public:
	FunctionOptions(Function& function) : m_function(function) {}

	constexpr static unsigned int DynamicBlockSize = 0;

	unsigned int GetBlockSize() const { return m_blockSize; }
	void SetBlockSize(unsigned int size)
	{
		if (m_blockSize == size)
		{
			return;
		}

		// Check that the new configuration is compatible with the existing settings

		if (m_blockSize != DynamicBlockSize)
		{
			Utils::Logger::LogError("Block size " + ((size == DynamicBlockSize) ? "<dynamic>" : std::to_string(size)) + " incompatible with block size " + std::to_string(m_blockSize));
		}
		else if (m_threadMultiple != 0 && (size % m_threadMultiple) != 0)
		{
			Utils::Logger::LogError("Block size " + ((size == DynamicBlockSize) ? "<dynamic>" : std::to_string(size)) + " incompatible with thread multiple " + std::to_string(m_threadMultiple));
		}

		m_blockSize = size;
	}

	unsigned int GetThreadMultiple() const { return m_threadMultiple; }
	void SetThreadMultiple(unsigned int multiple)
	{
		if (m_threadMultiple == multiple)
		{
			return;
		}

		if (m_threadMultiple == 0)
		{
			m_threadMultiple = multiple;
			return;
		}

		// Check that the new configuration is compatible with the existing settings
		// We choose the maximum multiple between the old/new provided that the larger
		// is a multiple of the smaller 

		if (multiple > m_threadMultiple && (multiple % m_threadMultiple) == 0)
		{
			if (m_blockSize != DynamicBlockSize && (m_blockSize % multiple) != 0)
			{
				Utils::Logger::LogError("Thread multiple " + std::to_string(multiple) + " incompatible with block size " + std::to_string(m_blockSize));
			}
			m_threadMultiple = multiple;
		}
		else if (multiple < m_threadMultiple && (m_threadMultiple % multiple) == 0)
		{
			return;
		}
		else
		{
			Utils::Logger::LogError("Thread multiple " + std::to_string(multiple) + " incompatible with thread multiple " + std::to_string(m_threadMultiple));
		}
	}

	unsigned int GetDynamicSharedMemorySize() const { return m_dynamicSharedMemorySize; }
	void SetDynamicSharedMemorySize(unsigned int bytes) { m_dynamicSharedMemorySize = bytes; }

	void SetCodegenOptions(const Codegen::InputOptions *codegenOptions) { m_codegenOptions = codegenOptions; }
	const Codegen::InputOptions *GetCodegenOptions() const { return m_codegenOptions; }

	std::string ToString() const
	{
		std::string output;
		output += "Block size: " + ((m_blockSize == DynamicBlockSize) ? "<dynamic>" : std::to_string(m_blockSize)) + "\n";
		output += "Thread multiple: " + std::to_string(m_threadMultiple) + "\n";
		output += "Shared memory size: " + std::to_string(m_dynamicSharedMemorySize) + " bytes";
		return output;
	}

	json ToJSON() const
	{
		json j;
		j["block_size"] = (m_blockSize == DynamicBlockSize) ? "<dynamic>" : std::to_string(m_blockSize);
		j["thread_multiple"] = std::to_string(m_threadMultiple);
		j["shared_memory"] = m_dynamicSharedMemorySize;
		return j;
	}

private:
	Function& m_function;

	unsigned int m_blockSize = DynamicBlockSize;
	unsigned int m_threadMultiple = 0;
	unsigned int m_dynamicSharedMemorySize = 0;

	const Codegen::InputOptions *m_codegenOptions = nullptr;
};

}
