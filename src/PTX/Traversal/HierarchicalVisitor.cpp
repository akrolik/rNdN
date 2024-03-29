#include "PTX/Traversal/HierarchicalVisitor.h"

#include "PTX/Tree/Tree.h"

namespace PTX {

// Node superclass

bool HierarchicalVisitor::VisitIn(Node *node)
{
	return true;
}

void HierarchicalVisitor::VisitOut(Node *node)
{

}

// Structural elements

bool HierarchicalVisitor::VisitIn(Program *program)
{
	return VisitIn(static_cast<Node*>(program));
}

bool HierarchicalVisitor::VisitIn(Module *module)
{
	return VisitIn(static_cast<Node*>(module));
}

bool HierarchicalVisitor::VisitIn(BasicBlock *block)
{
	return VisitIn(static_cast<Node*>(block));
}

void HierarchicalVisitor::VisitOut(Program *program)
{
	VisitOut(static_cast<Node*>(program));
}

void HierarchicalVisitor::VisitOut(Module *module)
{
	VisitOut(static_cast<Node*>(module));
}

void HierarchicalVisitor::VisitOut(BasicBlock *block)
{
	VisitOut(static_cast<Node*>(block));
}

// Functions

bool HierarchicalVisitor::VisitIn(Function *function)
{
	return VisitIn(static_cast<Node*>(function));
}

void HierarchicalVisitor::VisitOut(Function *function)
{
	VisitOut(static_cast<Node*>(function));
}

// Declarations

bool HierarchicalVisitor::VisitIn(Declaration *declaration)
{
	return VisitIn(static_cast<Node*>(declaration));
}

bool HierarchicalVisitor::VisitIn(VariableDeclaration *declaration)
{
	return VisitIn(static_cast<Declaration*>(declaration));
}

void HierarchicalVisitor::VisitOut(Declaration *declaration)
{
	VisitOut(static_cast<Node*>(declaration));
}

void HierarchicalVisitor::VisitOut(VariableDeclaration *declaration)
{
	VisitOut(static_cast<Declaration*>(declaration));
}

// Directives

bool HierarchicalVisitor::VisitIn(Directive *directive)
{
	return VisitIn(static_cast<Node*>(directive));
}

bool HierarchicalVisitor::VisitIn(FileDirective *directive)
{
	return VisitIn(static_cast<Directive*>(directive));
}

bool HierarchicalVisitor::VisitIn(LocationDirective *directive)
{
	return VisitIn(static_cast<Directive*>(directive));
}

void HierarchicalVisitor::VisitOut(Directive *directive)
{
	VisitOut(static_cast<Node*>(directive));
}

void HierarchicalVisitor::VisitOut(FileDirective *directive)
{
	VisitOut(static_cast<Directive*>(directive));
}

void HierarchicalVisitor::VisitOut(LocationDirective *directive)
{
	VisitOut(static_cast<Directive*>(directive));
}

// Instructions

bool HierarchicalVisitor::VisitIn(Statement *statement)
{
	return VisitIn(static_cast<Node*>(statement));
}

bool HierarchicalVisitor::VisitIn(BlockStatement *statement)
{
	return VisitIn(static_cast<Statement*>(statement));
}

bool HierarchicalVisitor::VisitIn(CommentStatement *statement)
{
	return VisitIn(static_cast<Statement*>(statement));
}

bool HierarchicalVisitor::VisitIn(DeclarationStatement *statement)
{
	return VisitIn(static_cast<Statement*>(statement));
}

bool HierarchicalVisitor::VisitIn(DirectiveStatement *statement)
{
	return VisitIn(static_cast<Statement*>(statement));
}

bool HierarchicalVisitor::VisitIn(InstructionStatement *statement)
{
	return VisitIn(static_cast<Statement*>(statement));
}

bool HierarchicalVisitor::VisitIn(LabelStatement *statement)
{
	return VisitIn(static_cast<Statement*>(statement));
}

void HierarchicalVisitor::VisitOut(Statement *statement)
{
	VisitOut(static_cast<Node*>(statement));
}

void HierarchicalVisitor::VisitOut(BlockStatement *statement)
{
	VisitOut(static_cast<Statement*>(statement));
}

void HierarchicalVisitor::VisitOut(CommentStatement *statement)
{
	VisitOut(static_cast<Statement*>(statement));
}

void HierarchicalVisitor::VisitOut(DeclarationStatement *statement)
{
	VisitOut(static_cast<Statement*>(statement));
}

void HierarchicalVisitor::VisitOut(DirectiveStatement *statement)
{
	VisitOut(static_cast<Statement*>(statement));
}

void HierarchicalVisitor::VisitOut(InstructionStatement *statement)
{
	VisitOut(static_cast<Statement*>(statement));
}

void HierarchicalVisitor::VisitOut(LabelStatement *statement)
{
	VisitOut(static_cast<Statement*>(statement));
}

// Instructions

bool HierarchicalVisitor::VisitIn(PredicatedInstruction *instruction)
{
	return VisitIn(static_cast<InstructionStatement*>(instruction));
}

void HierarchicalVisitor::VisitOut(PredicatedInstruction *instruction)
{
	VisitOut(static_cast<InstructionStatement*>(instruction));
}

bool HierarchicalVisitor::VisitIn(BranchInstruction *instruction)
{
	return VisitIn(static_cast<PredicatedInstruction*>(instruction));
}

void HierarchicalVisitor::VisitOut(BranchInstruction *instruction)
{
	VisitOut(static_cast<PredicatedInstruction*>(instruction));
}

bool HierarchicalVisitor::VisitIn(ReturnInstruction *instruction)
{
	return VisitIn(static_cast<PredicatedInstruction*>(instruction));
}

void HierarchicalVisitor::VisitOut(ReturnInstruction *instruction)
{
	VisitOut(static_cast<PredicatedInstruction*>(instruction));
}

// Operands

bool HierarchicalVisitor::VisitIn(Operand *operand)
{
	return VisitIn(static_cast<Node*>(operand));
}

void HierarchicalVisitor::VisitOut(Operand *operand)
{
	VisitOut(static_cast<Node*>(operand));
}

}
