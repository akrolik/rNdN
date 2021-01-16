#pragma once

namespace PTX {

class Node;

class Program;
class Module;
class BasicBlock;

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

class ConstHierarchicalVisitor
{
public:
	// Node superclass

	virtual bool VisitIn(const Node *node);
	virtual void VisitOut(const Node *node);

	// Structure

	virtual bool VisitIn(const Program *program);
	virtual bool VisitIn(const Module *module);
	virtual bool VisitIn(const BasicBlock *block);

	virtual void VisitOut(const Program *program);
	virtual void VisitOut(const Module *module);
	virtual void VisitOut(const BasicBlock *block);

	// Functions

	virtual bool VisitIn(const Function *function);
	virtual bool VisitIn(const FunctionDeclaration<VoidType> *function);
	virtual bool VisitIn(const FunctionDefinition<VoidType> *function);

	virtual void VisitOut(const Function *function);
	virtual void VisitOut(const FunctionDeclaration<VoidType> *function);
	virtual void VisitOut(const FunctionDefinition<VoidType> *function);

	// Declarations

	virtual bool VisitIn(const Declaration *declaration);
	virtual bool VisitIn(const VariableDeclaration *declaration); // Dispatch

	virtual void VisitOut(const Declaration *declaration);
	virtual void VisitOut(const VariableDeclaration *declaration); // Dispatch

	// Directives

	virtual bool VisitIn(const Directive *directive);
	virtual bool VisitIn(const FileDirective *directive);
	virtual bool VisitIn(const LocationDirective *directive);

	virtual void VisitOut(const Directive *directive);
	virtual void VisitOut(const FileDirective *directive);
	virtual void VisitOut(const LocationDirective *directive);

	// Statements

	virtual bool VisitIn(const Statement *statement);
	virtual bool VisitIn(const BlockStatement *statement);
	virtual bool VisitIn(const CommentStatement *statement);
	virtual bool VisitIn(const DeclarationStatement *statement);
	virtual bool VisitIn(const DirectiveStatement *statement);
	virtual bool VisitIn(const InstructionStatement *statement); // Dispatch
	virtual bool VisitIn(const LabelStatement *statement);

	virtual void VisitOut(const Statement *statement);
	virtual void VisitOut(const BlockStatement *statement);
	virtual void VisitOut(const CommentStatement *statement);
	virtual void VisitOut(const DeclarationStatement *statement);
	virtual void VisitOut(const DirectiveStatement *statement);
	virtual void VisitOut(const InstructionStatement *statement); // Dispatch
	virtual void VisitOut(const LabelStatement *statement);

	// Operands

	virtual bool VisitIn(const Operand *operand); // Dispatch
	virtual void VisitOut(const Operand *operand); // Dispatch
};

}
