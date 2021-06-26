#pragma once

#include "PTX/Analysis/ControlFlow/StructuredGraph/StructuredGraph.h"
#include "PTX/Analysis/ControlFlow/StructuredGraph/StructuredGraphVisitor.h"

#include "PTX/Tree/Tree.h"
#include "PTX/Traversal/ConstVisitor.h"
#include "PTX/Traversal/ConstInstructionVisitor.h"
#include "PTX/Traversal/ConstOperandVisitor.h"

namespace PTX {
namespace Transformation {

class BranchInliner : public Analysis::StructuredGraphVisitor, public ConstVisitor, public ConstInstructionVisitor, public ConstOperandVisitor
{
public:
	Analysis::StructureNode *Optimize(FunctionDefinition<VoidType> *function);

	// Structure visitor

	void Visit(Analysis::BranchStructure *structure) override;
	void Visit(Analysis::ExitStructure *structure) override;
	void Visit(Analysis::LoopStructure *structure) override;
	void Visit(Analysis::SequenceStructure *structure) override;
	void Visit(Analysis::StructureNode *structure) override;

	// Instructions visitor

	void Visit(const InstructionStatement *instruction) override;
	void Visit(const PredicatedInstruction *instruction) override;

	void Visit(const _MoveSpecialInstruction *instruction) override;
	void Visit(const _RemainderInstruction *instruction) override;
	void Visit(const _DivideInstruction *instruction) override;

	template<class T>
	void Visit(const MoveSpecialInstruction<T> *instruction);
	template<class T>
	void Visit(const RemainderInstruction<T> *instruction);
	template<class T>
	void Visit(const DivideInstruction<T> *instruction);

	// Operands visitor

	bool Visit(const _Value *value) override;
	bool Visit(const _Register *reg) override;

	template<class T>
	void Visit(const Register<T> *reg);
	template<class T>
	void Visit(const Value<T> *reg);

private:
	Analysis::SequenceStructure *m_sequenceNode = nullptr;
	Analysis::StructureNode *m_nextNode = nullptr;
	BasicBlock *m_nextBlock = nullptr;

	robin_hood::unordered_set<Analysis::ExitStructure *> m_exitStructures;
	Analysis::StructureNode *m_latchStructure = nullptr;

	const Register<PredicateType> *m_predicate = nullptr;
	bool m_predicated = false;
	bool m_power2 = false;
};

}
}
