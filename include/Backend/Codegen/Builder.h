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

	void SetCRSStackSize(std::size_t size);

	// Basic Blocks

	SASS::BasicBlock *CreateBasicBlock(const std::string& name);
	void CloseBasicBlock();

	// Instructions

	void AddInstruction(SASS::Instruction *instruction);

	// Temporary Registers

	SASS::Predicate *AllocateTemporaryPredicate()
	{
		//TODO: Allocate temporary predicate
		SASS::Predicate *pred = new SASS::Predicate(6);
		return pred;
	}

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
		
		// Insert data into constant memory

		auto offset = m_constantMemory.size();
		auto bytes = reinterpret_cast<const unsigned char *>(&data);
		for (auto i = 0; i < sizeof(T); ++i)
		{
			m_constantMemory.push_back(bytes[i]);
		}
		return offset;
	}

private:
	SASS::Function *m_currentFunction = nullptr;
	SASS::BasicBlock *m_currentBlock = nullptr;

	const PTX::Analysis::RegisterAllocation *m_registerAllocation = nullptr;
	const PTX::Analysis::ParameterSpaceAllocation *m_parameterSpaceAllocation = nullptr;

	std::uint8_t m_temporaryCount = 0;
	std::uint8_t m_temporaryMax = 0;

	std::vector<char> m_constantMemory;
};

}
}
