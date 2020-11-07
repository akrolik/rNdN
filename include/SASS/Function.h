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
	
	// Function elements

	const std::string& GetName() const { return m_name; }

	void AddParameter(std::size_t parameter) { m_parameters.push_back(parameter); }
	const std::vector<std::size_t>& GetParameters() const { return m_parameters; }
	std::size_t GetParametersCount() const { return m_parameters.size(); }

	void AddBasicBlock(BasicBlock *block) { m_blocks.push_back(block); }
	const std::vector<BasicBlock *>& GetBasicBlocks() const { return m_blocks; }

	void SetRegisters(const std::size_t registers) { m_registers = registers; }
	std::size_t GetRegisters() const { return m_registers; }

	// Format

	std::string ToString() const override
	{
		std::string code = ".text." + m_name + ":\n";
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
};

};
