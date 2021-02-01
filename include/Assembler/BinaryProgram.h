#pragma once

#include <vector>

#include "Assembler/BinaryFunction.h"

namespace Assembler {

class BinaryProgram
{
public:
	// Compute capability

	unsigned int GetComputeCapability() const { return m_computeCapability; }
	void SetComputeCapability(unsigned int computeCapability) { m_computeCapability = computeCapability; }

	// Shared memory

	bool GetDynamicSharedMemory() const { return m_dynamicSharedMemory; }
	void SetDynamicSharedMemory(bool dynamicSharedMemory) { m_dynamicSharedMemory = dynamicSharedMemory; }

	// Functions

	std::vector<const BinaryFunction *> GetFunctions() const
	{
		return { std::begin(m_functions), std::end(m_functions) };
	}
	std::vector<BinaryFunction *>& GetFunctions() { return m_functions; }

	unsigned int GetFunctionCount() const { return m_functions.size(); }

	void AddFunction(BinaryFunction *function) { m_functions.push_back(function); }
	void SetFunctions(const std::vector<BinaryFunction *>& functions) { m_functions = functions; }

private:
	unsigned int m_computeCapability = 0;
	bool m_dynamicSharedMemory = false;
	std::vector<BinaryFunction *> m_functions;
};

}
