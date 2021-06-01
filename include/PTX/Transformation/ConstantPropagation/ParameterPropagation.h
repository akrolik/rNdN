#pragma once

#include "PTX/Traversal/ConstInstructionVisitor.h"
#include "PTX/Traversal/ConstOperandVisitor.h"
#include "PTX/Traversal/InstructionVisitor.h"
#include "PTX/Traversal/Visitor.h"

#include "PTX/Tree/Tree.h"

#include "PTX/Analysis/BasicFlow/DefinitionsAnalysis.h"

namespace PTX {
namespace Transformation {

class ParameterPropagation : public Visitor, public ConstInstructionVisitor, public InstructionVisitor, public ConstOperandVisitor
{
public:
	ParameterPropagation(const Analysis::DefinitionsAnalysis& definitions) : m_definitions(definitions) {}

	void Transform(FunctionDefinition<VoidType> *function);

	// Structural visitors

	void Visit(FunctionDefinition<VoidType> *function) override;
	void Visit(BasicBlock *block) override;
	void Visit(InstructionStatement *instruction) override;

	// Instruction visitors

	void Visit(const _LoadInstruction *instruction) override;
	void Visit(const _ConvertToAddressInstruction *instruction) override;

	template<Bits B, class T, class S, LoadSynchronization A>
	void Visit(const LoadInstruction<B, T, S, A> *instruction);

	template<Bits B, class T, class S>
	void Visit(const ConvertToAddressInstruction<B, T, S> *instruction);

	// Transformation visitors

	void Visit(_MADWideInstruction *instruction) override;

	template<class T>
	void Visit(MADWideInstruction<T> *instruction);

	// Operand visitors

	bool Visit(const _Register *reg) override;

	template<class T>
	void Visit(const Register<T> *reg);
private:
	const Analysis::DefinitionsAnalysis& m_definitions;

	Operand *m_constantOperand = nullptr;
};

}
}
