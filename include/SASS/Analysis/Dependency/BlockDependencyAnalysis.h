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
	inline const static std::string Name = "Block dependency analysis";
	inline const static std::string ShortName = "dep";

	// Public API

	BlockDependencyAnalysis(const Function *function) : m_function(function) {}

	void Build(BasicBlock *block);
	const std::vector<BlockDependencyGraph *>& GetGraphs() const { return m_graphs; }

	// Visitors

	void Visit(Instruction *instruction) override;
	void Visit(PredicatedInstruction *instruction) override;

	void Visit(CS2RInstruction *instruction) override;
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
	void BuildDataDependencies(std::uint16_t operand);

	static const std::uint16_t DataOffset_Register  = 0;   // 256
	static const std::uint16_t DataOffset_Address   = 256; // 256
	static const std::uint16_t DataOffset_Carry     = 512; // 1
	static const std::uint16_t DataOffset_Predicate = 513; // 7

	std::vector<BlockDependencyGraph *> m_graphs;
	BlockDependencyGraph *m_graph = nullptr;

	robin_hood::unordered_map<std::uint16_t, std::vector<Instruction *>> m_readMap;
	robin_hood::unordered_map<std::uint16_t, std::vector<Instruction *>> m_writeMap;

	SASS::Instruction *m_instruction = nullptr;
	SASS::BasicBlock *m_block = nullptr;

	void InitializeSection();

	bool m_destination = false;
	bool m_predicated = false;

	const Function *m_function = nullptr;
};

}
}
