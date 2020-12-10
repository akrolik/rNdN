#pragma once

#include <string>
#include <sstream>

#include "PTX/Traversal/ConstVisitor.h"

namespace PTX {

class PrettyPrinter : public ConstVisitor
{
public:
	static std::string PrettyString(const Node *node);

	// Structure

	void Visit(const Program *program) override;
	void Visit(const Module *module) override;

	// Functions

	void Visit(const FunctionDeclaration<VoidType> *function) override;
	void Visit(const FunctionDefinition<VoidType> *function) override;

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

	bool m_definition = false;
};

}
