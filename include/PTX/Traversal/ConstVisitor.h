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

class ConstVisitor
{
public:
	// Node superclass

	virtual void Visit(const Node *node);

	// Structure

	virtual void Visit(const Program *program);
	virtual void Visit(const Module *module);

	// Functions

	virtual void Visit(const Function *function);
	virtual void Visit(const FunctionDeclaration<VoidType> *function);
	virtual void Visit(const FunctionDefinition<VoidType> *function);

	// Declarations

	virtual void Visit(const Declaration *declaration);
	virtual void Visit(const VariableDeclaration *declaration); // Dispatch

	// Directives

	virtual void Visit(const Directive *directive);
	virtual void Visit(const FileDirective *directive);
	virtual void Visit(const LocationDirective *directive);

	// Statements

	virtual void Visit(const Statement *statement);
	virtual void Visit(const BlockStatement *statement);
	virtual void Visit(const CommentStatement *statement);
	virtual void Visit(const DeclarationStatement *statement);
	virtual void Visit(const DirectiveStatement *statement);
	virtual void Visit(const InstructionStatement *statement); // Dispatch
	virtual void Visit(const LabelStatement *statement);

	// Operands

	virtual void Visit(const Operand *operand); // Dispatch
	virtual void Visit(const Label *label);
};

}
