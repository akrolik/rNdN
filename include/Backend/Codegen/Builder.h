#pragma once

#include "PTX/Analysis/RegisterAllocator/RegisterAllocation.h"
#include "PTX/Analysis/SpaceAllocator/GlobalSpaceAllocation.h"
#include "PTX/Analysis/SpaceAllocator/LocalSpaceAllocation.h"

#include "SASS/SASS.h"

#include "Utils/Math.h"

namespace Backend {
namespace Codegen {

class Builder
{
public:
	Builder(const PTX::Analysis::GlobalSpaceAllocation *allocation) : m_globalSpaceAllocation(allocation) {}

	// Register Allocation

	const PTX::Analysis::RegisterAllocation *GetRegisterAllocation() const { return m_registerAllocation; }
	void SetRegisterAllocation(const PTX::Analysis::RegisterAllocation *allocation) { m_registerAllocation = allocation; }

	// Space Allocation

	const PTX::Analysis::LocalSpaceAllocation *GetLocalSpaceAllocation() const { return m_localSpaceAllocation; }
	void SetLocalSpaceAllocation(const PTX::Analysis::LocalSpaceAllocation *allocation) { m_localSpaceAllocation = allocation; }

	const PTX::Analysis::GlobalSpaceAllocation *GetGlobalSpaceAllocation() const { return m_globalSpaceAllocation; }

	// Function

	SASS::Function *CreateFunction(const std::string& name);
	void CloseFunction();

	void AddParameter(std::size_t size);

	// Basic Blocks

	SASS::BasicBlock *CreateBasicBlock(const std::string& name);
	void CloseBasicBlock();

	// Instructions

	void AddInstruction(SASS::Instruction *instruction);

	// Temporary Registers

	SASS::Register *AllocateTemporaryRegister();
	void ClearTemporaryRegisters();

	// Relocations

	void AddRelocation(const SASS::Instruction *instruction, const std::string& name, SASS::Relocation::Kind kind);

	// Constant Memory

	template<class T>
	std::size_t AddConstantMemory(T data)
	{
		auto offset = m_constantMemory.size();
		auto bytes = reinterpret_cast<const unsigned char *>(&data);
		if constexpr(std::is_same<T, double>::value)
		{
			// Double values are word-reversed

			auto WORD_SIZE = 4;
			auto words = Utils::Math::DivUp(sizeof(T), WORD_SIZE);
			for (int i = words - 1; i >= 0; --i)
			{
				for (auto j = 0; j < WORD_SIZE && j < sizeof(T); ++j)
				{
					m_constantMemory.push_back(bytes[i * WORD_SIZE + j]);
				}
			}
		}
		else
		{
			for (auto i = 0; i < sizeof(T); ++i)
			{
				m_constantMemory.push_back(bytes[i]);
			}
		}
		return offset;
	}

private:
	SASS::Function *m_currentFunction = nullptr;
	SASS::BasicBlock *m_currentBlock = nullptr;

	const PTX::Analysis::RegisterAllocation *m_registerAllocation = nullptr;
	const PTX::Analysis::LocalSpaceAllocation *m_localSpaceAllocation = nullptr;
	const PTX::Analysis::GlobalSpaceAllocation *m_globalSpaceAllocation = nullptr;

	std::uint8_t m_temporaryCount = 0;
	std::uint8_t m_temporaryMax = 0;

	std::vector<char> m_constantMemory;
};

}
}
