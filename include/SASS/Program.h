#pragma once

#include <vector>

#include "SASS/Node.h"
#include "SASS/Function.h"

namespace SASS {

class Program : public Node
{
public:
	// Compute capability

	unsigned int GetComputeCapability() const { return m_computeCapability; }
	void SetComputeCapability(unsigned int computeCapability) { m_computeCapability = computeCapability; }

	// Shared memory

	bool GetDynamicSharedMemory() const { return m_dynamicSharedMemory; }
	void SetDynamicSharedMemory(bool dynamicSharedMemory) { m_dynamicSharedMemory = dynamicSharedMemory; }

	// Functions

	std::vector<const Function *> GetFunctions() const
	{
		return { std::begin(m_functions), std::end(m_functions) };
	}
	std::vector<Function *>& GetFunctions() { return m_functions; }

	void AddFunction(Function *function) { m_functions.push_back(function); }
	void SetFunctions(const std::vector<Function *>& functions) { m_functions = functions; }

	// Formatting

	std::string ToString() const override
	{
		std::string code;

		code += "// Program\n";
		code += "//  - Compute Capability: sm_" + std::to_string(m_computeCapability) + "\n";
		code += "//  - Dynamic Shared Memory: " + std::string(m_dynamicSharedMemory ? "True" : "False") + "\n\n";
		auto first = true;
		for (const auto& function : m_functions)
		{
			if (!first)
			{
				code += "\n\n";
			}
			first = false;
			code += function->ToString();
		}

		return code;
	}

private:
	unsigned int m_computeCapability = 0;
	bool m_dynamicSharedMemory = false;
	std::vector<Function *> m_functions;
};

};
