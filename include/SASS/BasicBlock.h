#pragma once

#include <vector>

#include "SASS/Node.h"
#include "SASS/Instructions/Instruction.h"

#include "Utils/Format.h"
#include "Utils/Logger.h"

namespace SASS {

class BasicBlock : public Node
{
public:
	BasicBlock(const std::string& name) : m_name(name) {}

	// Instructions

	std::vector<const Instruction *> GetInstructions() const
	{
		return { std::begin(m_instructions), std::end(m_instructions) };
	}
	std::vector<Instruction *>& GetInstructions() { return m_instructions; }

	void AddInstruction(Instruction *instruction) { m_instructions.push_back(instruction); }
	void SetInstructions(const std::vector<Instruction *>& instructions) { m_instructions = instructions; }

	// Name

	const std::string& GetName() const { return m_name; }
	void SetName(const std::string& name) { m_name = name; }

	// Formatting

	std::string ToString() const override
	{
		std::string code = "." + m_name + ":\n";
		for (auto instruction : m_instructions)
		{
			code += std::string(Utils::Logger::IndentSize, ' ');
			code += instruction->ToString() + "\n";
		}
		return code;
	}
	
private:
	std::string m_name;
	std::vector<Instruction *> m_instructions;
};

}
