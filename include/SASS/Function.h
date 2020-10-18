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

	const std::string& GetName() const { return m_name; }

	void AddBasicBlock(BasicBlock *block) { m_blocks.push_back(block); }
	const std::vector<BasicBlock *>& GetBasicBlocks() const { return m_blocks; }

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
	std::vector<BasicBlock *> m_blocks;
};

};
