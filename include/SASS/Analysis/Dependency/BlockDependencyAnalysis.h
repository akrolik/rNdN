#pragma once

#include "SASS/Analysis/Dependency/BlockDependencyGraph.h"
#include "SASS/Tree/Tree.h"

#include "SASS/Traversal/Visitor.h"

#include "Libraries/robin_hood.h"

namespace SASS {
namespace Analysis {

class BlockDependencyAnalysis : public Visitor
{
public:
	void Build(BasicBlock *block);

	BlockDependencyGraph *GetGraph() const { return m_graph; }

	// Visitors

	void Visit(Instruction *instruction) override;
	void Visit(PredicatedInstruction *instruction) override;

	void Visit(BRAInstruction *instruction) override;
	void Visit(BRKInstruction *instruction) override;
	void Visit(CONTInstruction *instruction) override;
	void Visit(EXITInstruction *instruction) override;
	void Visit(PBKInstruction *instruction) override;
	void Visit(PCNTInstruction *instruction) override;
	void Visit(RETInstruction *instruction) override;
	void Visit(SSYInstruction *instruction) override;
	void Visit(SYNCInstruction *instruction) override;
	void Visit(DEPBARInstruction *instruction) override;
	void Visit(BARInstruction *instruction) override;
	void Visit(MEMBARInstruction *instruction) override;

	void Visit(Address *address) override;
	void Visit(Register *reg) override;
	void Visit(Predicate *reg) override;
	void Visit(CarryFlag *carry) override;

private:
	void BuildControlDependencies(Instruction *controlInstruction);
	void BuildDataDependencies(std::uint32_t operand);

	static const std::uint32_t DataOffset_Register  = 0;   // 256
	static const std::uint32_t DataOffset_Address   = 256; // 256
	static const std::uint32_t DataOffset_Carry     = 512; // 1
	static const std::uint32_t DataOffset_Predicate = 513; // 7

	BlockDependencyGraph *m_graph = nullptr;

	robin_hood::unordered_map<std::uint32_t, robin_hood::unordered_set<Instruction *>> m_readMap;
	robin_hood::unordered_map<std::uint32_t, robin_hood::unordered_set<Instruction *>> m_writeMap;

	robin_hood::unordered_set<Instruction *> m_currentSet;

	SASS::Instruction *m_instruction = nullptr;
	SASS::Instruction *m_controlInstruction = nullptr;

	bool m_destination = false;
	bool m_predicated = false;
};

}
}
