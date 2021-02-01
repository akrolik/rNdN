#pragma once

#include <string>
#include <vector>

#include "SASS/Node.h"
#include "SASS/BasicBlock.h"

namespace SASS {

class Function : public Node
{
public:
	Function(const std::string& name) : m_name(name) {}
	
	// Function

	const std::string& GetName() const { return m_name; }
	void SetName(const std::string& name) { m_name = name; }

	// Parameters

	std::size_t GetParametersCount() const { return m_parameters.size(); }

	const std::vector<std::size_t>& GetParameters() const { return m_parameters; }
	std::vector<std::size_t>& GetParameters() { return m_parameters; }

	void AddParameter(std::size_t parameter) { m_parameters.push_back(parameter); }
	void SetParameters(const std::vector<std::size_t>& parameters) { m_parameters = parameters; }

	// Basic Blocks

	std::vector<const BasicBlock *> GetBasicBlocks() const
	{
		return { std::begin(m_blocks), std::end(m_blocks) };
	}
	std::vector<BasicBlock *>& GetBasicBlocks() { return m_blocks; }

	void AddBasicBlock(BasicBlock *block) { m_blocks.push_back(block); }
	void SetBasicBlocks(const std::vector<BasicBlock *>& blocks) { m_blocks = blocks; }

	// Registers

	void SetRegisters(const std::size_t registers) { m_registers = registers; }
	std::size_t GetRegisters() const { return m_registers; }

	// Threads
	
	std::tuple<std::size_t, std::size_t, std::size_t> GetRequiredThreads() const { return m_requiredThreads; }
	void SetRequiredThreads(std::size_t dimX, std::size_t dimY = 1, std::size_t dimZ = 1) { m_requiredThreads = { dimX, dimY, dimZ }; }

	std::tuple<std::size_t, std::size_t, std::size_t> GetMaxThreads() const { return m_maxThreads; }
	void SetMaxThreads(std::size_t dimX, std::size_t dimY = 1, std::size_t dimZ = 1) { m_maxThreads = { dimX, dimY, dimZ }; }

	// CTAID Z Dimension

	bool GetCTAIDZUsed() const { return m_ctaidzUsed; }
	void SetCTAIDZUsed(bool ctaidzUsed) { m_ctaidzUsed = ctaidzUsed; }

	// Shared Memory

	std::size_t GetSharedMemorySize() const { return m_sharedMemorySize; }
	void SetSharedMemorySize(std::size_t size) { m_sharedMemorySize = size; }

	// Formatting

	std::string ToString() const override
	{
		std::string code = "// " + m_name + "\n";
		if (m_parameters.size() > 0)
		{
			code += "// - Parameters (bytes): ";
			auto first = true;
			for (const auto parameter : m_parameters)
			{
				if (!first)
				{
					code += ", ";
				}
				first = false;
				code += std::to_string(parameter);
			}
			code += "\n";
		}
		code += "// - Registers: " + std::to_string(m_registers) + "\n";
		if (auto [dimX, dimY, dimZ] = m_requiredThreads; dimX > 0)
		{
			code += "// - Required Threads: " + std::to_string(dimX) + ", " + std::to_string(dimY) + ", " + std::to_string(dimZ) + "\n";
		}
		if (auto [dimX, dimY, dimZ] = m_maxThreads; dimX > 0)
		{
			code += "// - Max Threads: " + std::to_string(dimX) + ", " + std::to_string(dimY) + ", " + std::to_string(dimZ) + "\n";
		}
		code += "// - CTAIDZ Used: " + std::string((m_ctaidzUsed) ? "True" : "False") + "\n";
		code += "// - Shared Memory: " + std::to_string(m_sharedMemorySize) + " bytes\n";
		code += ".text." + m_name + ":\n";
		for (const auto& block : m_blocks)
		{
			code += block->ToString();
		}
		return code;
	}

private:
	std::string m_name;
	std::vector<std::size_t> m_parameters;
	std::vector<BasicBlock *> m_blocks;
	std::size_t m_registers = 0;

	std::tuple<std::size_t, std::size_t, std::size_t> m_requiredThreads;
	std::tuple<std::size_t, std::size_t, std::size_t> m_maxThreads;

	bool m_ctaidzUsed = false;

	std::size_t m_sharedMemorySize = 0;
};

};
