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

class BranchInstruction;
class ReturnInstruction;
class PredicatedInstruction;

class Operand;

class HierarchicalVisitor
{
public:
	// Node superclass

	virtual bool VisitIn(Node *node);
	virtual void VisitOut(Node *node);

	// Structure

	virtual bool VisitIn(Program *program);
	virtual bool VisitIn(Module *module);
	virtual bool VisitIn(BasicBlock *block);

	virtual void VisitOut(Program *program);
	virtual void VisitOut(Module *module);
	virtual void VisitOut(BasicBlock *block);

	// Functions

	virtual bool VisitIn(Function *function); // Dispatch
	virtual void VisitOut(Function *function); // Dispatch

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

	// Instructions

	virtual bool VisitIn(PredicatedInstruction *instruction);
	virtual void VisitOut(PredicatedInstruction *instruction);

	virtual bool VisitIn(BranchInstruction *instruction);
	virtual void VisitOut(BranchInstruction *instruction);

	virtual bool VisitIn(ReturnInstruction *instruction);
	virtual void VisitOut(ReturnInstruction *instruction);

	// Operands

	virtual bool VisitIn(Operand *operand); // Dispatch
	virtual void VisitOut(Operand *operand); // Dispatch
};

}
