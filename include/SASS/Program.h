#pragma once

#include <vector>

#include "SASS/Node.h"
#include "SASS/Function.h"
#include "SASS/GlobalVariable.h"

namespace SASS {

class Program : public Node
{
public:
	// Compute capability

	unsigned int GetComputeCapability() const { return m_computeCapability; }
	void SetComputeCapability(unsigned int computeCapability) { m_computeCapability = computeCapability; }

	// Global memory

	std::vector<const GlobalVariable *> GetGlobalVariables() const
	{
		return { std::begin(m_globalVariables), std::end(m_globalVariables) };
	}
	std::vector<GlobalVariable *>& GetGlobalVariables() { return m_globalVariables; }

	void AddGlobalVariable(GlobalVariable *globalVariable) { m_globalVariables.push_back(globalVariable); }
	void SetGlobalVariables(const std::vector<GlobalVariable *>& globalVariables) { m_globalVariables = globalVariables; }

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

		code += "// SASS Program\n";
		code += "//  - Compute Capability: sm_" + std::to_string(m_computeCapability) + "\n";
		code += "//  - Dynamic Shared Memory: " + std::string(m_dynamicSharedMemory ? "True" : "False");
		
		for (const auto& global : m_globalVariables)
		{
			code += "\n" + global->ToString();
		}
		for (const auto& function : m_functions)
		{
			code += "\n" + function->ToString();
		}

		return code;
	}

private:
	unsigned int m_computeCapability = 0;
	bool m_dynamicSharedMemory = false;
	std::vector<Function *> m_functions;
	std::vector<GlobalVariable *> m_globalVariables;
};

};
