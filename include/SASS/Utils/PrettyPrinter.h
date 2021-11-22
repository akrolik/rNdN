#pragma once

#include <string>
#include <sstream>

#include "SASS/Traversal/ConstVisitor.h"

namespace SASS {

class PrettyPrinter : public ConstVisitor
{
public:
	static std::string PrettyString(const Node *node, bool schedule = false);

	// Structure

	void Visit(const Program *program) override;
	void Visit(const Function *function) override;
	void Visit(const BasicBlock *block) override;

	void Visit(const Variable *variable) override;
	void Visit(const GlobalVariable *variable) override;
	void Visit(const SharedVariable *variable) override;
	void Visit(const DynamicSharedVariable *variable) override;

	void Visit(const Relocation *relocation) override;
	void Visit(const IndirectBranch *branch) override;

	// Instructions

	void Visit(const Instruction *instruction) override;

	void Visit(const Maxwell::PredicatedInstruction *instruction) override;
	void Visit(const Volta::PredicatedInstruction *instruction) override;

protected:
	void Indent();

	template<class I>
	void VisitPredicatedInstruction(const I *instruction);

	std::stringstream m_string;
	unsigned int m_indent = 0;

	bool m_schedule = false;
	bool m_predicated = false;
};

}
