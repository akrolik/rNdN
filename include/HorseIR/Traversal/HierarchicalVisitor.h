#pragma once

namespace HorseIR {

class Node;

class Program;
class Module;
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

class HierarchicalVisitor
{
public:
	// Node superclass

	virtual bool VisitIn(Node *node);
	virtual void VisitOut(Node *node);

	// Modules

	virtual bool VisitIn(Program *program);
	virtual bool VisitIn(Module *module);
	virtual bool VisitIn(ModuleContent *moduleContent);
	virtual bool VisitIn(ImportDirective *import);
	virtual bool VisitIn(GlobalDeclaration *global);
	virtual bool VisitIn(FunctionDeclaration *function);
	virtual bool VisitIn(BuiltinFunction *function);
	virtual bool VisitIn(Function *function);
	virtual bool VisitIn(VariableDeclaration *declaration);
	virtual bool VisitIn(Parameter *parameter);

	virtual void VisitOut(Program *program);
	virtual void VisitOut(Module *module);
	virtual void VisitOut(ModuleContent *moduleContent);
	virtual void VisitOut(ImportDirective *import);
	virtual void VisitOut(GlobalDeclaration *global);
	virtual void VisitOut(FunctionDeclaration *function);
	virtual void VisitOut(BuiltinFunction *function);
	virtual void VisitOut(Function *function);
	virtual void VisitOut(VariableDeclaration *declaration);
	virtual void VisitOut(Parameter *parameter);

	// Statements

	virtual bool VisitIn(Statement *statement);
	virtual bool VisitIn(DeclarationStatement *declarationS);
	virtual bool VisitIn(AssignStatement *assignS);
	virtual bool VisitIn(ExpressionStatement *expressionS);
        virtual bool VisitIn(IfStatement *ifS);
	virtual bool VisitIn(WhileStatement *whileS);
	virtual bool VisitIn(RepeatStatement *repeatS);
	virtual bool VisitIn(BlockStatement *blockS);
	virtual bool VisitIn(ReturnStatement *returnS);
	virtual bool VisitIn(BreakStatement *breakS);
	virtual bool VisitIn(ContinueStatement *continueS);

	virtual void VisitOut(Statement *statement);
	virtual void VisitOut(DeclarationStatement *declarationS);
	virtual void VisitOut(AssignStatement *assignS);
	virtual void VisitOut(ExpressionStatement *expressionS);
	virtual void VisitOut(IfStatement *ifS);
	virtual void VisitOut(WhileStatement *whileS);
	virtual void VisitOut(RepeatStatement *repeatS);
	virtual void VisitOut(BlockStatement *blockS);
	virtual void VisitOut(ReturnStatement *returnS);
	virtual void VisitOut(BreakStatement *breakS);
	virtual void VisitOut(ContinueStatement *continueS);

	// Expressions

	virtual bool VisitIn(Expression *expression);
	virtual bool VisitIn(CallExpression *call);
	virtual bool VisitIn(CastExpression *cast);
	virtual bool VisitIn(Operand *operand);
	virtual bool VisitIn(Literal *literal);
	virtual bool VisitIn(Identifier *identifier);

	virtual void VisitOut(Expression *expression);
	virtual void VisitOut(CastExpression *cast);
	virtual void VisitOut(CallExpression *call);
	virtual void VisitOut(Operand *operand);
	virtual void VisitOut(Literal *literal);
	virtual void VisitOut(Identifier *identifier);

	// Literals

	virtual bool VisitIn(VectorLiteral *literal);
	virtual bool VisitIn(BooleanLiteral *literal);
	virtual bool VisitIn(CharLiteral *literal);
	virtual bool VisitIn(Int8Literal *literal);
	virtual bool VisitIn(Int16Literal *literal);
	virtual bool VisitIn(Int32Literal *literal);
	virtual bool VisitIn(Int64Literal *literal);
	virtual bool VisitIn(Float32Literal *literal);
	virtual bool VisitIn(Float64Literal *literal);
	virtual bool VisitIn(ComplexLiteral *literal);
	virtual bool VisitIn(StringLiteral *literal);
	virtual bool VisitIn(SymbolLiteral *literal);
	virtual bool VisitIn(DatetimeLiteral *literal);
	virtual bool VisitIn(MonthLiteral *literal);
	virtual bool VisitIn(DateLiteral *literal);
	virtual bool VisitIn(MinuteLiteral *literal);
	virtual bool VisitIn(SecondLiteral *literal);
	virtual bool VisitIn(TimeLiteral *literal);
	virtual bool VisitIn(FunctionLiteral *literal);

	virtual void VisitOut(VectorLiteral *literal);
	virtual void VisitOut(BooleanLiteral *literal);
	virtual void VisitOut(CharLiteral *literal);
	virtual void VisitOut(Int8Literal *literal);
	virtual void VisitOut(Int16Literal *literal);
	virtual void VisitOut(Int32Literal *literal);
	virtual void VisitOut(Int64Literal *literal);
	virtual void VisitOut(Float32Literal *literal);
	virtual void VisitOut(Float64Literal *literal);
	virtual void VisitOut(ComplexLiteral *literal);
	virtual void VisitOut(StringLiteral *literal);
	virtual void VisitOut(SymbolLiteral *literal);
	virtual void VisitOut(DatetimeLiteral *literal);
	virtual void VisitOut(MonthLiteral *literal);
	virtual void VisitOut(DateLiteral *literal);
	virtual void VisitOut(MinuteLiteral *literal);
	virtual void VisitOut(SecondLiteral *literal);
	virtual void VisitOut(TimeLiteral *literal);
	virtual void VisitOut(FunctionLiteral *literal);

	// Types

	virtual bool VisitIn(Type *type);
	virtual bool VisitIn(WildcardType *type);
	virtual bool VisitIn(BasicType *type);
	virtual bool VisitIn(FunctionType *type);
	virtual bool VisitIn(ListType *type);
	virtual bool VisitIn(DictionaryType *type);
	virtual bool VisitIn(EnumerationType *type);
	virtual bool VisitIn(TableType *type);
	virtual bool VisitIn(KeyedTableType *type);

	virtual void VisitOut(Type *type);
	virtual void VisitOut(WildcardType *type);
	virtual void VisitOut(BasicType *type);
	virtual void VisitOut(FunctionType *type);
	virtual void VisitOut(ListType *type);
	virtual void VisitOut(DictionaryType *type);
	virtual void VisitOut(EnumerationType *type);
	virtual void VisitOut(TableType *type);
	virtual void VisitOut(KeyedTableType *type);
};

}
