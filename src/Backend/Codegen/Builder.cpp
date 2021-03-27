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
	m_currentFunction->SetConstantMemory(m_constantMemory);

	m_currentFunction = nullptr;
	m_registerAllocation = nullptr;
	m_parameterSpaceAllocation = nullptr;
	m_constantMemory.clear();
}

void Builder::AddParameter(std::size_t size)
{
	m_currentFunction->AddParameter(size);
}

void Builder::AddSharedVariable(const std::string& name, std::size_t size, std::size_t dataSize)
{
	m_currentFunction->AddSharedVariable(new SASS::SharedVariable(name, size, dataSize));
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

SASS::Register *Builder::AllocateTemporaryRegister(unsigned int align)
{
	// Find next free register

	auto offset = m_registerAllocation->GetRegisterCount();
	auto allocation = m_temporaryCount + offset;

	// Align Register

	allocation = Utils::Math::DivUp(allocation, align) * align;

	// Check within range

	if (allocation >= PTX::Analysis::RegisterAllocation::MaxRegister)
	{
		Utils::Logger::LogError("Temporary register exceeded maximum register count (" + std::to_string(PTX::Analysis::RegisterAllocation::MaxRegister) + ") for function '" + m_currentFunction->GetName() + "'");
	}
	m_temporaryCount = 1 + allocation - offset;

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

// Relocations

void Builder::AddRelocation(const SASS::Instruction *instruction, const std::string& name, SASS::Relocation::Kind kind)
{
	m_currentFunction->AddRelocation(new SASS::Relocation(instruction, name, kind));
}

// Indirect Branches

void Builder::AddIndirectBranch(const SASS::Instruction *instruction, const std::string& name)
{
	m_currentFunction->AddIndirectBranch(new SASS::IndirectBranch(instruction, name));
}

}
}
