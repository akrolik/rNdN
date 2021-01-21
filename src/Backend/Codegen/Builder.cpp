#include "Backend/Codegen/Builder.h"

namespace Backend {
namespace Codegen {

SASS::Function *Builder::CreateFunction(const std::string& name, const PTX::Analysis::RegisterAllocation *allocation)
{
	m_registerAllocation = allocation;
	m_currentFunction = new SASS::Function(name);
	m_currentFunction->SetRegisters(m_registerAllocation->GetRegisterCount());
	return m_currentFunction;
}

void Builder::CloseFunction()
{
	m_currentFunction = nullptr;
}

void Builder::AddParameter(std::size_t parameter)
{
	m_currentFunction->AddParameter(parameter);
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
