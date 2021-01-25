#include "Backend/Codegen/Builder.h"

namespace Backend {
namespace Codegen {

SASS::Function *Builder::CreateFunction(const std::string& name, const PTX::Analysis::RegisterAllocation *allocation)
{
	// Initialize allocators

	m_registerAllocation = allocation;
	m_parameterAllocation.clear();
	m_parameterOffset = SASS::CBANK_ParametersOffset;

	// Create function

	m_currentFunction = new SASS::Function(name);
	m_currentFunction->SetRegisters(m_registerAllocation->GetRegisterCount());

	return m_currentFunction;
}

void Builder::CloseFunction()
{
	m_currentFunction = nullptr;
	m_registerAllocation = nullptr;
	m_parameterAllocation.clear();
	m_parameterOffset = 0x0;
}

void Builder::AddParameter(const std::string& name, std::size_t size)
{
	m_parameterAllocation[name] = m_parameterOffset;
	m_parameterOffset += size;
	m_currentFunction->AddParameter(size);
}

std::size_t Builder::GetParameter(const std::string& name) const
{
	return m_parameterAllocation.at(name);
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
