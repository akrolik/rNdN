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
	struct Variable {
		std::string Name;
		std::size_t Size;
		std::size_t DataSize;
	};

	// Compute capability

	unsigned int GetComputeCapability() const { return m_computeCapability; }
	void SetComputeCapability(unsigned int computeCapability) { m_computeCapability = computeCapability; }

	// Global memory

	const std::vector<Variable>& GetGlobalVariables() const { return m_globalVariables; }
	std::size_t GetGlobalVariableCount() const { return m_globalVariables.size(); }

	void AddGlobalVariable(const std::string& name, std::size_t size, std::size_t dataSize)
	{
		m_globalVariables.push_back({ name, size, dataSize });
	}
	void SetGlobalVariables(const std::vector<Variable>& globalVariables) { m_globalVariables = globalVariables; }

	// Shared memory

	const std::vector<Variable>& GetSharedVariables() const { return m_sharedVariables; }
	std::size_t GetSharedVariableCount() const { return m_sharedVariables.size(); }

	void AddSharedVariable(const std::string& name, std::size_t size, std::size_t dataSize)
	{
		m_sharedVariables.push_back({ name, size, dataSize });
	}
	void SetSharedVariables(const std::vector<Variable>& sharedVariables) { m_sharedVariables = sharedVariables; }

	// Dynamic shared memory

	const std::vector<std::string>& GetDynamicSharedVariables() const { return m_dynamicSharedVariables; }
	std::size_t GetDynamicSharedVariableCount() const { return m_dynamicSharedVariables.size(); }

	void AddDynamicSharedVariable(const std::string& name) { m_dynamicSharedVariables.push_back(name); }
	void SetDynamicSharedVariables(const std::vector<std::string>& sharedVariables) { m_dynamicSharedVariables = sharedVariables; }

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
		for (const auto& variable : m_globalVariables)
		{
			code += "\n.global " + variable.Name + " { ";
			code += "size=" + Utils::Format::HexString(variable.Size) + " bytes; ";
			code += "datasize=" + Utils::Format::HexString(variable.DataSize) + " bytes }";
		}
		for (const auto& variable : m_sharedVariables)
		{
			code += "\n.shared " + variable.Name + " { ";
			code += "size=" + Utils::Format::HexString(variable.Size) + " bytes; ";
			code += "datasize=" + Utils::Format::HexString(variable.DataSize) + " bytes }";
		}
		for (const auto& variable : m_dynamicSharedVariables)
		{
			code += "\n.extern .shared " + variable;
		}
		for (const auto& function : m_functions)
		{
			code += "\n" + function->ToString();
		}
		return code;
	}

private:
	unsigned int m_computeCapability = 0;

	std::vector<Variable> m_globalVariables;
	std::vector<Variable> m_sharedVariables;
	std::vector<std::string> m_dynamicSharedVariables;

	std::vector<BinaryFunction *> m_functions;
};

}
