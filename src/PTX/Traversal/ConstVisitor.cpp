#include "PTX/Traversal/ConstVisitor.h"

#include "PTX/Tree/Tree.h"

namespace PTX {

// Node superclass

void ConstVisitor::Visit(const Node *node)
{

}

// Structural elements

void ConstVisitor::Visit(const Program *program)
{
	Visit(static_cast<const Node*>(program));
}

void ConstVisitor::Visit(const Module *module)
{
	Visit(static_cast<const Node*>(module));
}

void ConstVisitor::Visit(const BasicBlock *block)
{
	Visit(static_cast<const Node*>(block));
}

// Functions

void ConstVisitor::Visit(const Function *function)
{
	Visit(static_cast<const Node*>(function));
}

void ConstVisitor::Visit(const FunctionDeclaration<VoidType> *function)
{
	Visit(static_cast<const Function*>(function));
}

void ConstVisitor::Visit(const FunctionDefinition<VoidType> *function)
{
	Visit(static_cast<const FunctionDeclaration<VoidType>*>(function));
}

// Declarations

void ConstVisitor::Visit(const Declaration *declaration)
{
	Visit(static_cast<const Node*>(declaration));
}

void ConstVisitor::Visit(const VariableDeclaration *declaration)
{
	Visit(static_cast<const Declaration*>(declaration));
}

// Directives

void ConstVisitor::Visit(const Directive *directive)
{
	Visit(static_cast<const Node*>(directive));
}

void ConstVisitor::Visit(const FileDirective *directive)
{
	Visit(static_cast<const Directive*>(directive));
}

void ConstVisitor::Visit(const LocationDirective *directive)
{
	Visit(static_cast<const Directive*>(directive));
}

// Instructions

void ConstVisitor::Visit(const Statement *statement)
{
	Visit(static_cast<const Node*>(statement));
}

void ConstVisitor::Visit(const BlockStatement *statement)
{
	Visit(static_cast<const Statement*>(statement));
}

void ConstVisitor::Visit(const CommentStatement *statement)
{
	Visit(static_cast<const Statement*>(statement));
}

void ConstVisitor::Visit(const DeclarationStatement *statement)
{
	Visit(static_cast<const Statement*>(statement));
}

void ConstVisitor::Visit(const DirectiveStatement *statement)
{
	Visit(static_cast<const Statement*>(statement));
}

void ConstVisitor::Visit(const InstructionStatement *statement)
{
	Visit(static_cast<const Statement*>(statement));
}

void ConstVisitor::Visit(const LabelStatement *statement)
{
	Visit(static_cast<const Statement*>(statement));
}

// Operands

void ConstVisitor::Visit(const Operand *operand)
{
	Visit(static_cast<const Node*>(operand));
}

}
