#pragma once

#include "PTX/Analysis/RegisterAllocator/RegisterAllocation.h"
#include "PTX/Analysis/SpaceAllocator/SpaceAllocation.h"

#include "SASS/SASS.h"

namespace Backend {
namespace Codegen {

class Builder
{
public:
	// Register Allocation

	const PTX::Analysis::RegisterAllocation *GetRegisterAllocation() const { return m_registerAllocation; }
	void SetRegisterAllocation(const PTX::Analysis::RegisterAllocation *allocation) { m_registerAllocation = allocation; }

	// Space Allocation

	const PTX::Analysis::SpaceAllocation *GetSpaceAllocation() const { return m_spaceAllocation; }
	void SetSpaceAllocation(const PTX::Analysis::SpaceAllocation *allocation) { m_spaceAllocation = allocation; }

	// Function

	SASS::Function *CreateFunction(const std::string& name);
	void CloseFunction();

	void AddParameter(std::size_t size);

	// Basic Blocks

	SASS::BasicBlock *CreateBasicBlock(const std::string& name);
	void CloseBasicBlock();

	// Instructions

	void AddInstruction(SASS::Instruction *instruction);

	// Temporary Registers

	SASS::Register *AllocateTemporaryRegister();
	void ClearTemporaryRegisters();

private:
	SASS::Function *m_currentFunction = nullptr;
	SASS::BasicBlock *m_currentBlock = nullptr;

	const PTX::Analysis::RegisterAllocation *m_registerAllocation = nullptr;
	const PTX::Analysis::SpaceAllocation *m_spaceAllocation = nullptr;

	std::uint8_t m_temporaryCount = 0;
	std::uint8_t m_temporaryMax = 0;
};

}
}
