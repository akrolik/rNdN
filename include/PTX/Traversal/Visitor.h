#pragma once

namespace PTX {

class Node;

class Program;
class Module;

// Type used for functions
class VoidType;
class Function;
template<class T> class FunctionDeclaration;
template<class T> class FunctionDefinition;

class Declaration;
class VariableDeclaration;

class Directive;
class FileDirective;
class LocationDirective;

class Statement;
class BlockStatement;
class CommentStatement;
class DeclarationStatement;
class DirectiveStatement;
class InstructionStatement;
class LabelStatement;

class Operand;
class Label;

class Visitor
{
public:
	// Node superclass

	virtual void Visit(Node *node);

	// Structure

	virtual void Visit(Program *program);
	virtual void Visit(Module *module);

	// Functions

	virtual void Visit(Function *function);
	virtual void Visit(FunctionDeclaration<VoidType> *function);
	virtual void Visit(FunctionDefinition<VoidType> *function);

	// Declarations

	virtual void Visit(Declaration *declaration);
	virtual void Visit(VariableDeclaration *declaration); // Dispatch

	// Directives

	virtual void Visit(Directive *directive);
	virtual void Visit(FileDirective *directive);
	virtual void Visit(LocationDirective *directive);

	// Statements

	virtual void Visit(Statement *statement);
	virtual void Visit(BlockStatement *statement);
	virtual void Visit(CommentStatement *statement);
	virtual void Visit(DeclarationStatement *statement);
	virtual void Visit(DirectiveStatement *statement);
	virtual void Visit(InstructionStatement *statement); // Dispatch
	virtual void Visit(LabelStatement *statement);

	// Operands

	virtual void Visit(Operand *operand); // Dispatch
	virtual void Visit(Label *label);
};

}