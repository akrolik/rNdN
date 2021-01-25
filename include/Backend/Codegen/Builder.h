#pragma once

#include "PTX/Analysis/RegisterAllocator/RegisterAllocation.h"

#include "SASS/SASS.h"

#include <unordered_map>

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

	void AddParameter(const std::string& name, std::size_t size);
	std::size_t GetParameter(const std::string& name) const;

	// Basic Blocks

	SASS::BasicBlock *CreateBasicBlock(const std::string& name);
	void CloseBasicBlock();

	// Instructions

	void AddInstruction(SASS::Instruction *instruction);

private:
	SASS::Function *m_currentFunction = nullptr;
	SASS::BasicBlock *m_currentBlock = nullptr;

	std::unordered_map<std::string, std::size_t> m_parameterAllocation;
	std::size_t m_parameterOffset = 0;

	const PTX::Analysis::RegisterAllocation *m_registerAllocation = nullptr;
};

}
}
