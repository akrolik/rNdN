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

bool ConstHierarchicalVisitor::VisitIn(const BasicBlock *block)
{
	return VisitIn(static_cast<const Node*>(block));
}

void ConstHierarchicalVisitor::VisitOut(const Program *program)
{
	VisitOut(static_cast<const Node*>(program));
}

void ConstHierarchicalVisitor::VisitOut(const Module *module)
{
	VisitOut(static_cast<const Node*>(module));
}

void ConstHierarchicalVisitor::VisitOut(const BasicBlock *block)
{
	VisitOut(static_cast<const Node*>(block));
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

// Directives

bool ConstHierarchicalVisitor::VisitIn(const Directive *directive)
{
	return VisitIn(static_cast<const Node*>(directive));
}

bool ConstHierarchicalVisitor::VisitIn(const FileDirective *directive)
{
	return VisitIn(static_cast<const Directive*>(directive));
}

bool ConstHierarchicalVisitor::VisitIn(const LocationDirective *directive)
{
	return VisitIn(static_cast<const Directive*>(directive));
}

void ConstHierarchicalVisitor::VisitOut(const Directive *directive)
{
	VisitOut(static_cast<const Node*>(directive));
}

void ConstHierarchicalVisitor::VisitOut(const FileDirective *directive)
{
	VisitOut(static_cast<const Directive*>(directive));
}

void ConstHierarchicalVisitor::VisitOut(const LocationDirective *directive)
{
	VisitOut(static_cast<const Directive*>(directive));
}

// Instructions

bool ConstHierarchicalVisitor::VisitIn(const Statement *statement)
{
	return VisitIn(static_cast<const Node*>(statement));
}

bool ConstHierarchicalVisitor::VisitIn(const BlockStatement *statement)
{
	return VisitIn(static_cast<const Statement*>(statement));
}

bool ConstHierarchicalVisitor::VisitIn(const CommentStatement *statement)
{
	return VisitIn(static_cast<const Statement*>(statement));
}

bool ConstHierarchicalVisitor::VisitIn(const DeclarationStatement *statement)
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

bool ConstHierarchicalVisitor::VisitIn(const LabelStatement *statement)
{
	return VisitIn(static_cast<const Statement*>(statement));
}

void ConstHierarchicalVisitor::VisitOut(const Statement *statement)
{
	VisitOut(static_cast<const Node*>(statement));
}

void ConstHierarchicalVisitor::VisitOut(const BlockStatement *statement)
{
	VisitOut(static_cast<const Statement*>(statement));
}

void ConstHierarchicalVisitor::VisitOut(const CommentStatement *statement)
{
	VisitOut(static_cast<const Statement*>(statement));
}

void ConstHierarchicalVisitor::VisitOut(const DeclarationStatement *statement)
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

void ConstHierarchicalVisitor::VisitOut(const LabelStatement *statement)
{
	VisitOut(static_cast<const Statement*>(statement));
}

// Instructions

bool ConstHierarchicalVisitor::VisitIn(const PredicatedInstruction *instruction)
{
	return VisitIn(static_cast<const InstructionStatement*>(instruction));
}

void ConstHierarchicalVisitor::VisitOut(const PredicatedInstruction *instruction)
{
	VisitOut(static_cast<const InstructionStatement*>(instruction));
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
