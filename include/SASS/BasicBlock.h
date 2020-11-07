#pragma once

#include <vector>

#include "SASS/Node.h"
#include "SASS/Instructions/Instruction.h"

#include "Utils/Format.h"

namespace SASS {

class BasicBlock : public Node
{
public:
	BasicBlock(const std::string& name) : m_name(name) {}

	void AddInstruction(Instruction *instruction) { m_instructions.push_back(instruction); }
	const std::vector<Instruction *>& GetInstructions() const { return m_instructions; }

	const std::string& GetName() const { return m_name; }

	std::string ToString() const override
	{
		std::string code = "." + m_name + ":\n";
		for (auto instruction : m_instructions)
		{
			code += "\t" + instruction->ToString() + "\n";
		}
		return code;
	}
	
private:
	std::string m_name;
	std::vector<Instruction *> m_instructions;
};

}
