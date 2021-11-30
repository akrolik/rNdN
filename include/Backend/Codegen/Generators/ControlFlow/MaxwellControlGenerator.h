#pragma once

#include "Backend/Codegen/Generators/ControlFlow/ControlFlowGenerator.h"

#include "PTX/Tree/Tree.h"
#include "SASS/Tree/Tree.h"

namespace Backend {
namespace Codegen {

class MaxwellControlGenerator : public ControlFlowGenerator
{
public:
	using ControlFlowGenerator::ControlFlowGenerator;

	std::string Name() const override { return "MaxwellControlGenerator"; }

	void Generate(const PTX::FunctionDefinition<PTX::VoidType> *function) override;

	// Structure

	void Visit(const PTX::Analysis::StructureNode *structure) override;
	void Visit(const PTX::Analysis::BranchStructure *structure) override;
	void Visit(const PTX::Analysis::ExitStructure *structure) override;
	void Visit(const PTX::Analysis::LoopStructure *structure) override;
	void Visit(const PTX::Analysis::SequenceStructure *structure) override;

	// Basic block

	void VisitOut(const PTX::BasicBlock *block) override;
private:
	SASS::BasicBlock *m_beginBlock = nullptr;
	SASS::BasicBlock *m_endBlock = nullptr;

	std::vector<SASS::Instruction *> m_loopExits;
	std::size_t m_stackSize = 0;
	std::size_t m_maxStack = 0;
};

}
}
