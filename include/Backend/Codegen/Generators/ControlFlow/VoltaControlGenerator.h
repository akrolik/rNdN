#pragma once

#include "Backend/Codegen/Generators/ControlFlow/ControlFlowGenerator.h"

#include "PTX/Tree/Tree.h"
#include "SASS/Tree/Tree.h"

namespace Backend {
namespace Codegen {

class VoltaControlGenerator : public ControlFlowGenerator
{
public:
	using ControlFlowGenerator::ControlFlowGenerator;

	std::string Name() const override { return "VoltaControlGenerator"; }

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

	std::vector<SASS::Volta::BRAInstruction *> m_loopExits;
	std::size_t m_stackDepth = 0;
};

}
}
