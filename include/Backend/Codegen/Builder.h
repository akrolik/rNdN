#pragma once

#include "PTX/Analysis/RegisterAllocator/RegisterAllocation.h"
#include "PTX/Analysis/SpaceAllocator/ParameterSpaceAllocation.h"

#include "PTX/Tree/Tree.h"

#include "SASS/Tree/Tree.h"

#include "Utils/Math.h"

namespace Backend {
namespace Codegen {

class Builder
{
public:
	Builder(unsigned int computeCapability) : m_computeCapability(computeCapability) {}

	// Compute capability

	unsigned int GetComputeCapability() const { return m_computeCapability; }
	void SetComputeCapability(unsigned int computeCapability) { m_computeCapability = computeCapability; }

	// Register Allocation

	const PTX::Analysis::RegisterAllocation *GetRegisterAllocation() const { return m_registerAllocation; }
	void SetRegisterAllocation(const PTX::Analysis::RegisterAllocation *allocation) { m_registerAllocation = allocation; }

	// Space Allocation

	const PTX::Analysis::ParameterSpaceAllocation *GetParameterSpaceAllocation() const { return m_parameterSpaceAllocation; }
	void SetParameterSpaceAllocation(const PTX::Analysis::ParameterSpaceAllocation *allocation) { m_parameterSpaceAllocation = allocation; }

	// Function

	SASS::Function *CreateFunction(const std::string& name);
	void CloseFunction();

	void AddParameter(std::size_t size);
	void AddSharedVariable(const std::string& name, std::size_t size, std::size_t dataSize);

	void SetMaxThreads(const std::tuple<unsigned int, unsigned int, unsigned int>& threads);
	void SetRequiredThreads(const std::tuple<unsigned int, unsigned int, unsigned int>& threads);

	void SetCRSStackSize(std::size_t size);

	// Basic Blocks

	SASS::BasicBlock *CreateBasicBlock(const std::string& name);
	void CloseBasicBlock();

	SASS::BasicBlock *GetCurrentBlock() { return m_currentBlock; }

	// Instructions

	void AddInstruction(SASS::Instruction *instruction);

	// Temporary Registers

	SASS::Predicate *AllocateTemporaryPredicate();

	template<PTX::Bits B>
	std::pair<SASS::Register *, SASS::Register *> AllocateTemporaryRegisterPair()
	{
		auto align = Utils::Math::DivUp(PTX::BitSize<B>::NumBits, 32);
		return AllocateTemporaryRegisterPair<B>(align);
	}

	template<PTX::Bits B>
	std::pair<SASS::Register *, SASS::Register *> AllocateTemporaryRegisterPair(unsigned int align)
	{
		if constexpr(PTX::BitSize<B>::NumBits <= 32)
		{
			return { AllocateTemporaryRegister(align), nullptr };
		}
		else
		{
			auto reg0 = AllocateTemporaryRegister(align);
			auto reg1 = AllocateTemporaryRegister();
			return { reg0, reg1 };
		}
	}

	template<PTX::Bits B>
	SASS::Register *AllocateTemporaryRegister()
	{
		// Size and alignment are the same

		auto size = Utils::Math::DivUp(PTX::BitSize<B>::NumBits, 32);
		return AllocateTemporaryRegister(size, size);
	}

	SASS::Register *AllocateTemporaryRegister(unsigned int align = 1, unsigned int range = 1);
	void ClearTemporaryRegisters();

	// Relocations

	void AddRelocation(const SASS::Instruction *instruction, const std::string& name, SASS::Relocation::Kind kind);

	// Indirect Branches

	void AddIndirectBranch(const SASS::Instruction *instruction, const std::string& name);

	// Constant Memory

	template<class T>
	std::size_t AddConstantMemory(T data)
	{
		// Must be aligned for the type

		auto align = m_constantMemory.size() % sizeof(T);
		if (align > 0)
		{
			auto adjust = sizeof(T) - align;
			for (auto i = 0; i < adjust; ++i)
			{
				m_constantMemory.push_back(0);
			}
		}

		if (sizeof(T) > m_constantMemoryAlign)
		{
			m_constantMemoryAlign = sizeof(T);
		}
		
		// Insert data into constant memory

		auto offset = m_constantMemory.size();
		auto bytes = reinterpret_cast<const unsigned char *>(&data);
		for (auto i = 0; i < sizeof(T); ++i)
		{
			m_constantMemory.push_back(bytes[i]);
		}
		return offset;
	}

	// Unique identifiers

	std::string UniqueIdentifier(const std::string& name)
	{
		return (name + "_" + std::to_string(m_uniqueIndex++));
	}

private:
	SASS::Function *m_currentFunction = nullptr;
	SASS::BasicBlock *m_currentBlock = nullptr;

	const PTX::Analysis::RegisterAllocation *m_registerAllocation = nullptr;
	const PTX::Analysis::ParameterSpaceAllocation *m_parameterSpaceAllocation = nullptr;

	std::uint8_t m_predicateCount = 0;
	std::uint8_t m_temporaryCount = 0;
	std::uint8_t m_temporaryMax = 0;

	std::vector<char> m_constantMemory;
	std::size_t m_constantMemoryAlign = 0;

	unsigned int m_uniqueIndex = 0;
	unsigned int m_computeCapability = 0;
};

}
}
