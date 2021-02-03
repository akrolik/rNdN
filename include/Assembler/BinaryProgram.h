#pragma once

#include <string>
#include <utility>
#include <vector>

#include "Assembler/BinaryFunction.h"

#include "Utils/Format.h"

namespace Assembler {

class BinaryProgram
{
public:
	// Compute capability

	unsigned int GetComputeCapability() const { return m_computeCapability; }
	void SetComputeCapability(unsigned int computeCapability) { m_computeCapability = computeCapability; }

	// Global memory

	const std::vector<std::tuple<std::string, std::size_t, std::size_t>>& GetGlobalVariables() const { return m_globalVariables; }
	std::size_t GetGlobalVariableCount() const { return m_globalVariables.size(); }

	void AddGlobalVariable(const std::string& name, std::size_t offset, std::size_t size) { m_globalVariables.push_back({ name, offset, size }); }
	void SetGlobalVariables(const std::vector<std::tuple<std::string, std::size_t, std::size_t>>& globalVariables) { m_globalVariables = globalVariables; }

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

	std::string ToString() const
	{
		std::string code;
		code += "// Binary SASS Program\n";
		code += "// - Compute Capability: sm_" + std::to_string(m_computeCapability) + "\n";
		code += "// - Dynamic Shared Memory: " + std::string(m_dynamicSharedMemory ? "True" : "False");
		for (const auto& [name, offset, size] : m_globalVariables)
		{
			code += "\n.global " + name + "(offset=" + Utils::Format::HexString(offset) + ", size=" + Utils::Format::HexString(size) + ")";
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
	std::vector<std::tuple<std::string, std::size_t, std::size_t>> m_globalVariables;
	std::vector<BinaryFunction *> m_functions;
};

}
