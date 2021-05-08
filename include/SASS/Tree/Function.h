#pragma once

#include <string>
#include <vector>

#include "SASS/Tree/Node.h"
#include "SASS/Tree/BasicBlock.h"
#include "SASS/Tree/Relocation.h"
#include "SASS/Tree/SharedVariable.h"
#include "SASS/Tree/IndirectBranch.h"

namespace SASS {

class Function : public Node
{
public:
	Function(const std::string& name) : m_name(name) {}
	
	// Function

	const std::string& GetName() const { return m_name; }
	void SetName(const std::string& name) { m_name = name; }

	// Parameters

	std::size_t GetParametersCount() const { return m_parameters.size(); }

	const std::vector<std::size_t>& GetParameters() const { return m_parameters; }
	std::vector<std::size_t>& GetParameters() { return m_parameters; }

	void AddParameter(std::size_t parameter) { m_parameters.push_back(parameter); }
	void SetParameters(const std::vector<std::size_t>& parameters) { m_parameters = parameters; }

	// Basic Blocks

	std::vector<const BasicBlock *> GetBasicBlocks() const
	{
		return { std::begin(m_blocks), std::end(m_blocks) };
	}
	std::vector<BasicBlock *>& GetBasicBlocks() { return m_blocks; }

	void AddBasicBlock(BasicBlock *block) { m_blocks.push_back(block); }
	void SetBasicBlocks(const std::vector<BasicBlock *>& blocks) { m_blocks = blocks; }

	// Registers

	void SetRegisters(const std::size_t registers) { m_registers = registers; }
	std::size_t GetRegisters() const { return m_registers; }

	// Threads
	
	const std::tuple<std::size_t, std::size_t, std::size_t>& GetRequiredThreads() const { return m_requiredThreads; }
	void SetRequiredThreads(std::size_t dimX, std::size_t dimY = 1, std::size_t dimZ = 1) { m_requiredThreads = { dimX, dimY, dimZ }; }

	const std::tuple<std::size_t, std::size_t, std::size_t>& GetMaxThreads() const { return m_maxThreads; }
	void SetMaxThreads(std::size_t dimX, std::size_t dimY = 1, std::size_t dimZ = 1) { m_maxThreads = { dimX, dimY, dimZ }; }

	// CTAID Z Dimension

	bool GetCTAIDZUsed() const { return m_ctaidzUsed; }
	void SetCTAIDZUsed(bool ctaidzUsed) { m_ctaidzUsed = ctaidzUsed; }

	// Shared Memory

	const std::vector<const SharedVariable *> GetSharedVariables() const
	{
		return { std::begin(m_sharedVariables), std::end(m_sharedVariables) };
	}
	std::vector<SharedVariable *>& GetSharedVariables() { return m_sharedVariables; }

	void AddSharedVariable(SharedVariable *sharedVariable) { m_sharedVariables.push_back(sharedVariable); }
	void SetSharedVariables(const std::vector<SharedVariable *>& sharedVariables) { m_sharedVariables = sharedVariables; }

	// Constant Memory

	std::size_t GetConstantMemorySize() const { return m_constantMemory.size(); }

	const std::vector<char>& GetConstantMemory() const { return m_constantMemory; }
	void SetConstantMemory(const std::vector<char>& constantMemory) { m_constantMemory = constantMemory; }

	// Relocations

	const std::vector<const Relocation *> GetRelocations() const
	{
		return { std::begin(m_relocations), std::end(m_relocations) };
	}
	std::vector<Relocation *>& GetRelocations() { return m_relocations; }

	void AddRelocation(Relocation *relocation) { m_relocations.push_back(relocation); }
	void SetRelocations(const std::vector<Relocation *>& relocations) { m_relocations = relocations; }

	// Indirect Branches

	const std::vector<const IndirectBranch *> GetIndirectBranches() const
	{
		return { std::begin(m_indirectBranches), std::end(m_indirectBranches) };
	}
	std::vector<IndirectBranch *>& GetIndirectBranches() { return m_indirectBranches; }

	void AddIndirectBranch(IndirectBranch *indirectBranch) { m_indirectBranches.push_back(indirectBranch); }
	void SetIndirectBranches(const std::vector<IndirectBranch *>& indirectBranches) { m_indirectBranches = indirectBranches; }

	// CRS Stack Size

	std::size_t GetCRSStackSize() const { return m_crsStackSize; }
	void SetCRSStackSize(std::size_t crsStackSize) { m_crsStackSize = crsStackSize; }

	// Visitors

	void Accept(Visitor& visitor) override { visitor.Visit(this); }
	void Accept(ConstVisitor& visitor) const override { visitor.Visit(this); }
	
private:
	std::string m_name;
	std::vector<std::size_t> m_parameters;
	std::vector<BasicBlock *> m_blocks;
	std::size_t m_registers = 0;
	std::size_t m_crsStackSize = 0;

	std::tuple<std::size_t, std::size_t, std::size_t> m_requiredThreads;
	std::tuple<std::size_t, std::size_t, std::size_t> m_maxThreads;

	bool m_ctaidzUsed = false;

	std::vector<SharedVariable *> m_sharedVariables;

	std::vector<char> m_constantMemory;

	std::vector<Relocation *> m_relocations;
	std::vector<IndirectBranch *> m_indirectBranches;
};

};
