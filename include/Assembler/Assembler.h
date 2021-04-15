#pragma once

#include <unordered_map>

#include "Assembler/BinaryFunction.h"
#include "Assembler/BinaryProgram.h"

#include "SASS/Tree/Tree.h"
#include "SASS/Traversal/Visitor.h"

namespace Assembler {

class Assembler : public SASS::Visitor
{
public:
	BinaryProgram *Assemble(const SASS::Program *program);
	BinaryFunction *Assemble(const SASS::Function *function);

	// Address resolution

	void Visit(SASS::DivergenceInstruction *instruction) override;
	void Visit(SASS::BRAInstruction *instruction) override;

	// NVIDIA custom ELF

	void Visit(SASS::EXITInstruction *instruction) override;
	void Visit(SASS::S2RInstruction *instruction) override;
	void Visit(SASS::SHFLInstruction *instruction) override;
	void Visit(SASS::BARInstruction *instruction) override;

protected:
	std::uint32_t m_index = 0;
	std::uint32_t m_barrierCount = 0;
	
	std::unordered_map<std::string, unsigned int> m_blockIndex;
	BinaryFunction *m_binaryFunction = nullptr;
};

}
