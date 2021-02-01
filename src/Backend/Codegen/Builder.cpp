#include "Backend/Codegen/Builder.h"

#include "Utils/Logger.h"

namespace Backend {
namespace Codegen {

SASS::Function *Builder::CreateFunction(const std::string& name)
{
	// Create function

	m_currentFunction = new SASS::Function(name);
	return m_currentFunction;
}

void Builder::CloseFunction()
{
	// Set register count

	m_currentFunction->SetRegisters(m_registerAllocation->GetRegisterCount() + m_temporaryMax);
	m_currentFunction->SetSharedMemorySize(m_localSpaceAllocation->GetSharedMemorySize() + m_globalSpaceAllocation->GetSharedMemorySize());

	m_currentFunction = nullptr;
	m_registerAllocation = nullptr;
	m_localSpaceAllocation = nullptr;
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

// Temporary Registers

SASS::Register *Builder::AllocateTemporaryRegister()
{
	// Find next free register

	auto offset = m_registerAllocation->GetRegisterCount();
	auto allocation = m_temporaryCount + offset;
	if (allocation >= PTX::Analysis::RegisterAllocation::MaxRegister)
	{
		Utils::Logger::LogError("Temporary register exceeded maximum register count (" + std::to_string(PTX::Analysis::RegisterAllocation::MaxRegister) + ") for function '" + m_currentFunction->GetName() + "'");
	}
	m_temporaryCount++;

	// Maintain high water mark for total registers

	if (m_temporaryCount > m_temporaryMax)
	{
		m_temporaryMax = m_temporaryCount;
	}
	return new SASS::Register(allocation);
}

void Builder::ClearTemporaryRegisters()
{
	m_temporaryCount = 0;
}

}
}
