#include "PTX/Traversal/Visitor.h"

#include "PTX/Tree/Tree.h"

namespace PTX {

// Node superclass

void Visitor::Visit(Node *node)
{

}

// Structural elements

void Visitor::Visit(Program *program)
{
	Visit(static_cast<Node*>(program));
}

void Visitor::Visit(Module *module)
{
	Visit(static_cast<Node*>(module));
}

void Visitor::Visit(BasicBlock *block)
{
	Visit(static_cast<Node*>(block));
}

// Functions

void Visitor::Visit(Function *function)
{
	Visit(static_cast<Node*>(function));
}

// Declarations

void Visitor::Visit(Declaration *declaration)
{
	Visit(static_cast<Node*>(declaration));
}

void Visitor::Visit(VariableDeclaration *declaration)
{
	Visit(static_cast<Declaration*>(declaration));
}

// Directives

void Visitor::Visit(Directive *directive)
{
	Visit(static_cast<Node*>(directive));
}

void Visitor::Visit(FileDirective *directive)
{
	Visit(static_cast<Directive*>(directive));
}

void Visitor::Visit(LocationDirective *directive)
{
	Visit(static_cast<Directive*>(directive));
}

// Instructions

void Visitor::Visit(Statement *statement)
{
	Visit(static_cast<Node*>(statement));
}

void Visitor::Visit(BlockStatement *statement)
{
	Visit(static_cast<Statement*>(statement));
}

void Visitor::Visit(CommentStatement *statement)
{
	Visit(static_cast<Statement*>(statement));
}

void Visitor::Visit(DeclarationStatement *statement)
{
	Visit(static_cast<Statement*>(statement));
}

void Visitor::Visit(DirectiveStatement *statement)
{
	Visit(static_cast<Statement*>(statement));
}

void Visitor::Visit(InstructionStatement *statement)
{
	Visit(static_cast<Statement*>(statement));
}

void Visitor::Visit(LabelStatement *statement)
{
	Visit(static_cast<Statement*>(statement));
}

// Instructions

void Visitor::Visit(PredicatedInstruction *instruction)
{
	Visit(static_cast<InstructionStatement*>(instruction));
}

// Operands

void Visitor::Visit(Operand *operand)
{
	Visit(static_cast<Node*>(operand));
}

}
