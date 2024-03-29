#pragma once

#include <string>
#include <sstream>

#include "PTX/Traversal/ConstVisitor.h"
#include "PTX/Traversal/ConstFunctionVisitor.h"
#include "PTX/Tree/Tree.h"

namespace PTX {

class PrettyPrinter : public ConstVisitor, public ConstFunctionVisitor
{
public:
	static std::string PrettyString(const Node *node, bool quick = false);

	// Structure

	void Visit(const Program *program) override;
	void Visit(const Module *module) override;
	void Visit(const BasicBlock *block) override;

	// Functions

	void Visit(const Function *function) override;

	void Visit(const _FunctionDeclaration *function) override;
	void Visit(const _FunctionDefinition *function) override;

	template<class T, class S>
	void Visit(const FunctionDeclaration<T, S> *function);
	template<class T, class S>
	void Visit(const FunctionDefinition<T, S> *function);

	// Declarations

	void Visit(const VariableDeclaration *declaration) override;

	// Directives

	void Visit(const FileDirective *directive) override;
	void Visit(const LocationDirective *directive) override;

	// Statements

	void Visit(const BlockStatement *statement) override;
	void Visit(const CommentStatement *statement) override;
	void Visit(const DeclarationStatement *statement) override;
	void Visit(const InstructionStatement *statement) override;
	void Visit(const LabelStatement *statement) override;

	// Operands

	void Visit(const Operand *operand) override;

protected:
	void Indent();

	std::stringstream m_string;
	unsigned int m_indent = 0;

	bool m_quick = false;
	bool m_definition = false;
};

}
