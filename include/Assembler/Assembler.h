#pragma once

#include "Assembler/BinaryFunction.h"
#include "Assembler/BinaryProgram.h"

#include "SASS/Tree/Tree.h"
#include "SASS/Traversal/Visitor.h"

#include "Libraries/robin_hood.h"

namespace Assembler {

class Assembler : public SASS::Visitor
{
public:
	BinaryProgram *Assemble(const SASS::Program *program);
	BinaryFunction *Assemble(const SASS::Function *function);

	// Address resolution

	std::uint64_t GetTargetAddress(const std::string& target) const;

	void Visit(SASS::Maxwell::DivergenceInstruction *instruction) override;
	void Visit(SASS::Volta::DivergenceInstruction *instruction) override;

	// NVIDIA custom ELF

	void Visit(SASS::Maxwell::EXITInstruction *instruction) override;
	void Visit(SASS::Maxwell::S2RInstruction *instruction) override;
	void Visit(SASS::Maxwell::SHFLInstruction *instruction) override;
	void Visit(SASS::Maxwell::BARInstruction *instruction) override;

	void Visit(SASS::Volta::EXITInstruction *instruction) override;
	void Visit(SASS::Volta::S2RInstruction *instruction) override;
	void Visit(SASS::Volta::SHFLInstruction *instruction) override;
	void Visit(SASS::Volta::BARInstruction *instruction) override;

protected:
	SASS::Instruction *GeneratePaddingInstruction() const;
	SASS::Instruction *GenerateSelfBranchInstruction(const std::string& name) const;

	BinaryFunction *m_binaryFunction = nullptr;
	unsigned int m_computeCapability = 0;

	std::uint8_t m_instructionPad = 0;
	std::uint8_t m_instructionSize = 0;
	std::uint8_t m_scheduleGroup = 0;

	std::uint32_t m_barrierCount = 0;
	std::uint64_t m_currentAddress = 0x0;
	robin_hood::unordered_map<std::string, std::uint64_t> m_blockAddress;
};

}
