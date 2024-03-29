#pragma once

namespace PTX {

class Node;

class Program;
class Module;
class BasicBlock;

class Function;

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

class PredicatedInstruction;

class Operand;

class ConstVisitor
{
public:
	// Node superclass

	virtual void Visit(const Node *node);

	// Structure

	virtual void Visit(const Program *program);
	virtual void Visit(const Module *module);
	virtual void Visit(const BasicBlock *block);

	// Functions

	virtual void Visit(const Function *function); // Dispatch

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

	// Instructions

	virtual void Visit(const PredicatedInstruction *instruction);

	// Operands

	virtual void Visit(const Operand *operand); // Dispatch
};

}
