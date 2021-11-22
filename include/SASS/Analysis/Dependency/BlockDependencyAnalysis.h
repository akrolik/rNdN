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

	BlockDependencyAnalysis(const Function *function) : m_function(function), m_node(m_tempNode), m_tempNode(nullptr) {}

	void Build(BasicBlock *block);
	const std::vector<BlockDependencyGraph *>& GetGraphs() const { return m_graphs; }

	// Visitors

	void Visit(Instruction *instruction) override;
	void Visit(Maxwell::PredicatedInstruction *instruction) override;
	void Visit(Volta::PredicatedInstruction *instruction) override;

	void Visit(Address *address) override;
	void Visit(Register *reg) override;
	void Visit(Predicate *reg) override;
	void Visit(CarryFlag *carry) override;

private:
	template<class T>
	void BuildPredicatedInstruction(T *instruction);

	void BuildControlDependencies(Instruction *controlInstruction);
	void BuildDataDependencies(std::uint16_t operand);

	static const std::uint16_t DataOffset_Register  = 0;   // 256
	static const std::uint16_t DataOffset_Address   = 256; // 256
	static const std::uint16_t DataOffset_Carry     = 512; // 1
	static const std::uint16_t DataOffset_Predicate = 513; // 7

	std::vector<BlockDependencyGraph *> m_graphs;
	BlockDependencyGraph *m_graph = nullptr;

	robin_hood::unordered_flat_map<std::uint16_t, std::vector<std::reference_wrapper<Analysis::BlockDependencyGraph::Node>>> m_readMap;
	robin_hood::unordered_flat_map<std::uint16_t, std::vector<std::reference_wrapper<Analysis::BlockDependencyGraph::Node>>> m_writeMap;

	BasicBlock *m_block = nullptr;
	Instruction *m_instruction = nullptr;

	std::reference_wrapper<Analysis::BlockDependencyGraph::Node> m_node;
	Analysis::BlockDependencyGraph::Node m_tempNode;

	void InitializeSection(bool control = false);

	bool m_destination = false;
	bool m_predicated = false;

	const Function *m_function = nullptr;
};

}
}
