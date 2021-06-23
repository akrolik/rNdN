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
	m_currentFunction->SetMaxRegisters(m_registerAllocation->GetMaxRegisters());
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

void Builder::SetMaxThreads(const std::tuple<unsigned int, unsigned int, unsigned int>& threads)
{
	m_currentFunction->SetMaxThreads(std::get<0>(threads), std::get<1>(threads), std::get<2>(threads));
}

void Builder::SetRequiredThreads(const std::tuple<unsigned int, unsigned int, unsigned int>& threads)
{
	m_currentFunction->SetRequiredThreads(std::get<0>(threads), std::get<1>(threads), std::get<2>(threads));
}

void Builder::SetCRSStackSize(std::size_t size)
{
	m_currentFunction->SetCRSStackSize(size);
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

SASS::Predicate *Builder::AllocateTemporaryPredicate()
{
	// Find next free predicate

	auto offset = m_registerAllocation->GetPredicateCount();
	auto allocation = m_predicateCount + offset;

	// Check within range

	if (auto maxPredicate = m_registerAllocation->GetMaxPredicates(); allocation >= maxPredicate)
	{
		Utils::Logger::LogError("Temporary predicate exceeded maximum predicate count (" + std::to_string(maxPredicate) + ") for function '" + m_currentFunction->GetName() + "'");
	}

	return new SASS::Predicate(allocation);
}

SASS::Register *Builder::AllocateTemporaryRegister(unsigned int align, unsigned int range)
{
	// Find next free register

	auto offset = m_registerAllocation->GetRegisterCount();
	auto allocation = m_temporaryCount + offset;

	// Align Register

	allocation = Utils::Math::DivUp(allocation, align) * align;

	// Check within range

	if (auto maxRegisters = m_registerAllocation->GetMaxRegisters(); allocation + (range - 1) >= maxRegisters)
	{
		Utils::Logger::LogError("Temporary register exceeded maximum register count (" + std::to_string(maxRegisters) + ") for function '" + m_currentFunction->GetName() + "'");
	}
	m_temporaryCount = range + allocation - offset;

	// Maintain high water mark for total registers

	if (m_temporaryCount > m_temporaryMax)
	{
		m_temporaryMax = m_temporaryCount;
	}

	return new SASS::Register(allocation, range);
}

void Builder::ClearTemporaryRegisters()
{
	m_temporaryCount = 0;
	m_predicateCount = 0;
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
