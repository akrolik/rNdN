#pragma once

namespace HorseIR {

class Node;

class Program;
class Module;
class LibraryModule;
class ModuleContent;
class ImportDirective;
class GlobalDeclaration;
class FunctionDeclaration;
class BuiltinFunction;
class Function;
class VariableDeclaration;
class Parameter;

class Statement;
class DeclarationStatement;
class AssignStatement;
class ExpressionStatement;
class IfStatement;
class WhileStatement;
class RepeatStatement;
class BlockStatement;
class ReturnStatement;
class BreakStatement;
class ContinueStatement;

class Expression;
class CallExpression;
class CastExpression;
class Operand;
class Identifier;

class Literal;
class VectorLiteral;
class BooleanLiteral;
class CharLiteral;
class Int8Literal;
class Int16Literal;
class Int32Literal;
class Int64Literal;
class Float32Literal;
class Float64Literal;
class ComplexLiteral;
class StringLiteral;
class SymbolLiteral;
class DatetimeLiteral;
class MonthLiteral;
class DateLiteral;
class MinuteLiteral;
class SecondLiteral;
class TimeLiteral;
class FunctionLiteral;

class Type;
class WildcardType;
class BasicType;
class FunctionType;
class ListType;
class DictionaryType;
class EnumerationType;
class TableType;
class KeyedTableType;

class Visitor
{
public:
	// Node superclass

	virtual void Visit(Node *node);

	// Modules

	virtual void Visit(Program *program);
	virtual void Visit(Module *module);
	virtual void Visit(LibraryModule *module);
	virtual void Visit(ModuleContent *moduleContent);
	virtual void Visit(ImportDirective *import);
	virtual void Visit(GlobalDeclaration *global);
	virtual void Visit(FunctionDeclaration *function);
	virtual void Visit(BuiltinFunction *function);
	virtual void Visit(Function *function);
	virtual void Visit(VariableDeclaration *declaration);
	virtual void Visit(Parameter *parameter);

	// Statements

	virtual void Visit(Statement *statement);
	virtual void Visit(DeclarationStatement *declarationS);
	virtual void Visit(AssignStatement *assignS);
	virtual void Visit(ExpressionStatement *expressionS);
        virtual void Visit(IfStatement *ifS);
	virtual void Visit(WhileStatement *whileS);
	virtual void Visit(RepeatStatement *repeatS);
	virtual void Visit(BlockStatement *blockS);
	virtual void Visit(ReturnStatement *returnS);
	virtual void Visit(BreakStatement *breakS);
	virtual void Visit(ContinueStatement *continueS);

	// Expressions

	virtual void Visit(Expression *expression);
	virtual void Visit(CallExpression *call);
	virtual void Visit(CastExpression *cast);
	virtual void Visit(Operand *operand);
	virtual void Visit(Literal *literal);
	virtual void Visit(Identifier *identifier);

	// Literals

	virtual void Visit(VectorLiteral *literal);
	virtual void Visit(BooleanLiteral *literal);
	virtual void Visit(CharLiteral *literal);
	virtual void Visit(Int8Literal *literal);
	virtual void Visit(Int16Literal *literal);
	virtual void Visit(Int32Literal *literal);
	virtual void Visit(Int64Literal *literal);
	virtual void Visit(Float32Literal *literal);
	virtual void Visit(Float64Literal *literal);
	virtual void Visit(ComplexLiteral *literal);
	virtual void Visit(StringLiteral *literal);
	virtual void Visit(SymbolLiteral *literal);
	virtual void Visit(DatetimeLiteral *literal);
	virtual void Visit(MonthLiteral *literal);
	virtual void Visit(DateLiteral *literal);
	virtual void Visit(MinuteLiteral *literal);
	virtual void Visit(SecondLiteral *literal);
	virtual void Visit(TimeLiteral *literal);
	virtual void Visit(FunctionLiteral *literal);

	// Types

	virtual void Visit(Type *type);
	virtual void Visit(WildcardType *type);
	virtual void Visit(BasicType *type);
	virtual void Visit(FunctionType *type);
	virtual void Visit(ListType *type);
	virtual void Visit(DictionaryType *type);
	virtual void Visit(EnumerationType *type);
	virtual void Visit(TableType *type);
	virtual void Visit(KeyedTableType *type);
};

}
