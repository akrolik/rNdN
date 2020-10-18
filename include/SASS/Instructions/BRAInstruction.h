#pragma once

#include "SASS/Instructions/BRAInstruction.h"
#include "SASS/BasicBlock.h"

namespace SASS {

class BRAInstruction : public Instruction
{
public:
	BRAInstruction(const BasicBlock *block) : m_block(block) {}

	std::string OpCode() const override { return "BRA"; }

	std::string ToString() const override
	{
		return OpCode() + " `(" + m_block->GetName() + ") ;";
	}

private:
	const BasicBlock *m_block = nullptr;
};

}
