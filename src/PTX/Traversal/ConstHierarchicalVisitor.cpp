#include "PTX/Traversal/ConstHierarchicalVisitor.h"

#include "PTX/Tree/Tree.h"

namespace PTX {

// Node superclass

bool ConstHierarchicalVisitor::VisitIn(const Node *node)
{
	return true;
}

void ConstHierarchicalVisitor::VisitOut(const Node *node)
{

}

// Structural elements

bool ConstHierarchicalVisitor::VisitIn(const Program *program)
{
	return VisitIn(static_cast<const Node*>(program));
}

bool ConstHierarchicalVisitor::VisitIn(const Module *module)
{
	return VisitIn(static_cast<const Node*>(module));
}

void ConstHierarchicalVisitor::VisitOut(const Program *program)
{
	VisitOut(static_cast<const Node*>(program));
}

void ConstHierarchicalVisitor::VisitOut(const Module *module)
{
	VisitOut(static_cast<const Node*>(module));
}

// Functions

bool ConstHierarchicalVisitor::VisitIn(const Function *function)
{
	return VisitIn(static_cast<const Node*>(function));
}

bool ConstHierarchicalVisitor::VisitIn(const FunctionDeclaration<VoidType> *function)
{
	return VisitIn(static_cast<const Function*>(function));
}

bool ConstHierarchicalVisitor::VisitIn(const FunctionDefinition<VoidType> *function)
{
	return VisitIn(static_cast<const FunctionDeclaration<VoidType>*>(function));
}

void ConstHierarchicalVisitor::VisitOut(const Function *function)
{
	VisitOut(static_cast<const Node*>(function));
}

void ConstHierarchicalVisitor::VisitOut(const FunctionDeclaration<VoidType> *function)
{
	VisitOut(static_cast<const Function*>(function));
}

void ConstHierarchicalVisitor::VisitOut(const FunctionDefinition<VoidType> *function)
{
	VisitOut(static_cast<const FunctionDeclaration<VoidType>*>(function));
}

// Declarations

bool ConstHierarchicalVisitor::VisitIn(const Declaration *declaration)
{
	return VisitIn(static_cast<const Node*>(declaration));
}

bool ConstHierarchicalVisitor::VisitIn(const VariableDeclaration *declaration)
{
	return VisitIn(static_cast<const Declaration*>(declaration));
}

void ConstHierarchicalVisitor::VisitOut(const Declaration *declaration)
{
	VisitOut(static_cast<const Node*>(declaration));
}

void ConstHierarchicalVisitor::VisitOut(const VariableDeclaration *declaration)
{
	VisitOut(static_cast<const Declaration*>(declaration));
}

// Instructions

bool ConstHierarchicalVisitor::VisitIn(const Statement *statement)
{
	return VisitIn(static_cast<const Node*>(statement));
}

bool ConstHierarchicalVisitor::VisitIn(const BlankStatement *statement)
{
	return VisitIn(static_cast<const Statement*>(statement));
}

bool ConstHierarchicalVisitor::VisitIn(const BlockStatement *statement)
{
	return VisitIn(static_cast<const Statement*>(statement));
}

bool ConstHierarchicalVisitor::VisitIn(const CommentStatement *statement)
{
	return VisitIn(static_cast<const Statement*>(statement));
}

bool ConstHierarchicalVisitor::VisitIn(const DirectiveStatement *statement)
{
	return VisitIn(static_cast<const Statement*>(statement));
}

bool ConstHierarchicalVisitor::VisitIn(const InstructionStatement *statement)
{
	return VisitIn(static_cast<const Statement*>(statement));
}

bool ConstHierarchicalVisitor::VisitIn(const Label *statement)
{
	return VisitIn(static_cast<const Statement*>(statement));
}

void ConstHierarchicalVisitor::VisitOut(const Statement *statement)
{
	VisitOut(static_cast<const Node*>(statement));
}

void ConstHierarchicalVisitor::VisitOut(const BlankStatement *statement)
{
	VisitOut(static_cast<const Statement*>(statement));
}

void ConstHierarchicalVisitor::VisitOut(const BlockStatement *statement)
{
	VisitOut(static_cast<const Statement*>(statement));
}

void ConstHierarchicalVisitor::VisitOut(const CommentStatement *statement)
{
	VisitOut(static_cast<const Statement*>(statement));
}

void ConstHierarchicalVisitor::VisitOut(const DirectiveStatement *statement)
{
	VisitOut(static_cast<const Statement*>(statement));
}

void ConstHierarchicalVisitor::VisitOut(const InstructionStatement *statement)
{
	VisitOut(static_cast<const Statement*>(statement));
}

void ConstHierarchicalVisitor::VisitOut(const Label *statement)
{
	VisitOut(static_cast<const Statement*>(statement));
}

// Operands

bool ConstHierarchicalVisitor::VisitIn(const Operand *operand)
{
	return VisitIn(static_cast<const Node*>(operand));
}

void ConstHierarchicalVisitor::VisitOut(const Operand *operand)
{
	VisitOut(static_cast<const Node*>(operand));
}

}
