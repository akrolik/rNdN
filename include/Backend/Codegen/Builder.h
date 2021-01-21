#pragma once

#include "PTX/Analysis/RegisterAllocator/RegisterAllocation.h"

#include "SASS/SASS.h"

namespace Backend {
namespace Codegen {

class Builder
{
public:
	// Register Allocation

	const PTX::Analysis::RegisterAllocation *GetRegisterAllocation() const { return m_registerAllocation; }
	void SetRegisterAllocation(const PTX::Analysis::RegisterAllocation *allocation) { m_registerAllocation = allocation; }

	// Function

	SASS::Function *CreateFunction(const std::string& name, const PTX::Analysis::RegisterAllocation *allocation);
	void CloseFunction();

	void AddParameter(std::size_t parameter);

	// Basic Blocks

	SASS::BasicBlock *CreateBasicBlock(const std::string& name);
	void CloseBasicBlock();

	// Instructions

	void AddInstruction(SASS::Instruction *instruction);

private:
	SASS::Function *m_currentFunction = nullptr;
	SASS::BasicBlock *m_currentBlock = nullptr;

	const PTX::Analysis::RegisterAllocation *m_registerAllocation = nullptr;
};

}
}
