#pragma once

#include "PTX/Traversal/ConstHierarchicalVisitor.h"
#include "PTX/Traversal/ConstDeclarationVisitor.h"

#include "PTX/Analysis/ControlFlow/StructuredGraph/ConstStructuredGraphVisitor.h"
#include "PTX/Analysis/ControlFlow/StructuredGraph/StructuredGraph.h"

#include "Backend/Codegen/Builder.h"

#include "PTX/Analysis/RegisterAllocator/RegisterAllocation.h"
#include "PTX/Analysis/SpaceAllocator/ParameterSpaceAllocation.h"
#include "PTX/Tree/Tree.h"

#include "SASS/Tree/Tree.h"

namespace Backend {
namespace Codegen {

class CodeGenerator : public PTX::ConstHierarchicalVisitor, public PTX::ConstDeclarationVisitor, public PTX::Analysis::ConstStructuredGraphVisitor
{
public:
	CodeGenerator(unsigned int computeCapability) : m_builder(computeCapability) {}

	// Generator

	SASS::Function *Generate(const PTX::FunctionDefinition<PTX::VoidType> *function, const PTX::Analysis::RegisterAllocation *registerAllocation, const PTX::Analysis::ParameterSpaceAllocation *parameterAllocation);
	
	// Declarations

	bool VisitIn(const PTX::VariableDeclaration *declaration) override;

	template<class T, class S>
	void Visit(const PTX::TypedVariableDeclaration<T, S> *declaration);
	void Visit(const PTX::_TypedVariableDeclaration *declaration) override;

	// Structure

	void Visit(const PTX::Analysis::StructureNode *structure) override;
	void Visit(const PTX::Analysis::BranchStructure *structure) override;
	void Visit(const PTX::Analysis::ExitStructure *structure) override;
	void Visit(const PTX::Analysis::LoopStructure *structure) override;
	void Visit(const PTX::Analysis::SequenceStructure *structure) override;

	// Basic Block
	
	bool VisitIn(const PTX::BasicBlock *block) override;
	void VisitOut(const PTX::BasicBlock *block) override;

	// Statements

	bool VisitIn(const PTX::InstructionStatement *statement) override;

	// Compute capability

	unsigned int GetComputeCapability() const { return m_builder.GetComputeCapability(); }
	void SetComputeCapability(unsigned int computeCapability) { m_builder.SetComputeCapability(computeCapability); }

private:
	template<class SSY, class SYNC, class BRA, bool BARRIER = false>
	void GenerateBranchStructure(const PTX::Analysis::BranchStructure *structure);

	template<class SYNC, bool BARRIER = false>
	void GenerateExitStructure(const PTX::Analysis::ExitStructure *structure);

	template<class SSY, class BRA, bool BARRIER = false>
	void GenerateLoopStructure(const PTX::Analysis::LoopStructure *structure);

	Builder m_builder;

	SASS::BasicBlock *m_beginBlock = nullptr;
	SASS::BasicBlock *m_endBlock = nullptr;

	std::vector<SASS::Instruction *> m_loopExits;
	std::size_t m_stackSize = 0;
	std::size_t m_maxStack = 0;
	std::size_t m_stackDepth = 0;
};

}
}
