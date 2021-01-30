#include "Backend/Codegen/Builder.h"

#include "Utils/Logger.h"

namespace Backend {
namespace Codegen {

SASS::Function *Builder::CreateFunction(const std::string& name)
{
	// Create function

	m_currentFunction = new SASS::Function(name);
	m_currentFunction->SetRegisters(m_registerAllocation->GetRegisterCount());

	return m_currentFunction;
}

void Builder::CloseFunction()
{
	m_currentFunction = nullptr;
	m_registerAllocation = nullptr;
	m_spaceAllocation = nullptr;
}

void Builder::AddParameter(std::size_t size)
{
	m_currentFunction->AddParameter(size);
}

// Basic Blocks

SASS::BasicBlock *Builder::CreateBasicBlock(const std::string& name)
{
	m_currentBlock = new SASS::BasicBlock(name);
	m_currentFunction->AddBasicBlock(m_currentBlock);
	return m_currentBlock;
}

void Builder::CloseBasicBlock()
{
	m_currentBlock = nullptr;
}

// Instructions

void Builder::AddInstruction(SASS::Instruction *instruction)
{
	m_currentBlock->AddInstruction(instruction);
}

}
}
