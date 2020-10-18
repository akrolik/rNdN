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

	const std::string& GetName() const { return m_name; }

	std::string ToString() const override
	{
		std::string code = "." + m_name + ":\n";
		for (auto i = 0u; i < m_instructions.size(); ++i)
		{
			if (i % 3 == 0 && i+2 < m_instructions.size())
			{
				auto inst1 = m_instructions.at(i);
				auto inst2 = m_instructions.at(i+1);
				auto inst3 = m_instructions.at(i+2);

				std::uint64_t assembled = 0u;
				assembled <<= 1;
				assembled |= inst3->GetScheduling().GenCode();
				assembled <<= 21;
				assembled |= inst2->GetScheduling().GenCode();
				assembled <<= 21;
				assembled |= inst1->GetScheduling().GenCode();

				code += "\t /* " + Utils::Format::HexString(assembled, 16) + " */\n";
			}
			auto instruction = m_instructions.at(i);
			code += "\t" + instruction->ToString() + "\n";
		}
		return code;
	}
	
private:
	std::string m_name;
	std::vector<Instruction *> m_instructions;
};

}
