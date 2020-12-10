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

class HierarchicalVisitor
{
public:
	// Node superclass

	virtual bool VisitIn(Node *node);
	virtual void VisitOut(Node *node);

	// Structure

	virtual bool VisitIn(Program *program);
	virtual bool VisitIn(Module *module);

	virtual void VisitOut(Program *program);
	virtual void VisitOut(Module *module);

	// Functions

	virtual bool VisitIn(Function *function);
	virtual bool VisitIn(FunctionDeclaration<VoidType> *function);
	virtual bool VisitIn(FunctionDefinition<VoidType> *function);

	virtual void VisitOut(Function *function);
	virtual void VisitOut(FunctionDeclaration<VoidType> *function);
	virtual void VisitOut(FunctionDefinition<VoidType> *function);

	// Declarations

	virtual bool VisitIn(Declaration *declaration);
	virtual bool VisitIn(VariableDeclaration *declaration); // Dispatch

	virtual void VisitOut(Declaration *declaration);
	virtual void VisitOut(VariableDeclaration *declaration); // Dispatch

	// Directives

	virtual bool VisitIn(Directive *directive);
	virtual bool VisitIn(FileDirective *directive);
	virtual bool VisitIn(LocationDirective *directive);

	virtual void VisitOut(Directive *directive);
	virtual void VisitOut(FileDirective *directive);
	virtual void VisitOut(LocationDirective *directive);

	// Statements

	virtual bool VisitIn(Statement *statement);
	virtual bool VisitIn(BlockStatement *statement);
	virtual bool VisitIn(CommentStatement *statement);
	virtual bool VisitIn(DeclarationStatement *statement);
	virtual bool VisitIn(DirectiveStatement *statement);
	virtual bool VisitIn(InstructionStatement *statement); // Dispatch
	virtual bool VisitIn(LabelStatement *statement);

	virtual void VisitOut(Statement *statement);
	virtual void VisitOut(BlockStatement *statement);
	virtual void VisitOut(CommentStatement *statement);
	virtual void VisitOut(DeclarationStatement *statement);
	virtual void VisitOut(DirectiveStatement *statement);
	virtual void VisitOut(InstructionStatement *statement); // Dispatch
	virtual void VisitOut(LabelStatement *statement);

	// Operands

	virtual bool VisitIn(Operand *operand); // Dispatch
	virtual bool VisitIn(Label *label);

	virtual void VisitOut(Operand *operand); // Dispatch
	virtual void VisitOut(Label *label);
};

}
