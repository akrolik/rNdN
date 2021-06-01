#pragma once

#include "PTX/Traversal/ConstInstructionVisitor.h"
#include "PTX/Traversal/Visitor.h"

#include "PTX/Tree/Tree.h"

#include "PTX/Analysis/BasicFlow/LiveVariables.h"

namespace PTX {
namespace Transformation {

class DeadCodeElimination : public Visitor, public ConstOperandDispatcher<DeadCodeElimination>
{
public:
	DeadCodeElimination(const Analysis::LiveVariables& liveVariables) : m_liveVariables(liveVariables) {}

	bool Transform(FunctionDefinition<VoidType> *function);

	// Structural visitors

	void Visit(FunctionDefinition<VoidType> *function) override;
	void Visit(BasicBlock *block) override;
	void Visit(InstructionStatement *instruction) override;

	// Operand visitors

	using ConstOperandDispatcher<DeadCodeElimination>::Visit;

	template<class T> void Visit(const Register<T> *reg);

private:
	const Analysis::LiveVariables& m_liveVariables;

	const InstructionStatement *m_currentStatement = nullptr;
	
	bool m_transform = false;
	bool m_dead = false;
};

}
}
