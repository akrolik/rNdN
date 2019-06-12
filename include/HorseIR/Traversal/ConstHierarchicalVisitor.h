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

class ConstHierarchicalVisitor
{
public:
	// Node superclass

	virtual bool VisitIn(const Node *node);
	virtual void VisitOut(const Node *node);

	// Modules

	virtual bool VisitIn(const Program *program);
	virtual bool VisitIn(const Module *module);
	virtual bool VisitIn(const ModuleContent *moduleContent);
	virtual bool VisitIn(const ImportDirective *import);
	virtual bool VisitIn(const GlobalDeclaration *global);
	virtual bool VisitIn(const FunctionDeclaration *function);
	virtual bool VisitIn(const BuiltinFunction *function);
	virtual bool VisitIn(const Function *function);
	virtual bool VisitIn(const VariableDeclaration *declaration);
	virtual bool VisitIn(const Parameter *parameter);

	virtual void VisitOut(const Program *program);
	virtual void VisitOut(const Module *module);
	virtual void VisitOut(const ModuleContent *moduleContent);
	virtual void VisitOut(const ImportDirective *import);
	virtual void VisitOut(const GlobalDeclaration *global);
	virtual void VisitOut(const FunctionDeclaration *function);
	virtual void VisitOut(const BuiltinFunction *function);
	virtual void VisitOut(const Function *function);
	virtual void VisitOut(const VariableDeclaration *declaration);
	virtual void VisitOut(const Parameter *parameter);

	// Statements

	virtual bool VisitIn(const Statement *statement);
	virtual bool VisitIn(const DeclarationStatement *declarationS);
	virtual bool VisitIn(const AssignStatement *assignS);
	virtual bool VisitIn(const ExpressionStatement *expressionS);
        virtual bool VisitIn(const IfStatement *ifS);
	virtual bool VisitIn(const WhileStatement *whileS);
	virtual bool VisitIn(const RepeatStatement *repeatS);
	virtual bool VisitIn(const BlockStatement *blockS);
	virtual bool VisitIn(const ReturnStatement *returnS);
	virtual bool VisitIn(const BreakStatement *breakS);
	virtual bool VisitIn(const ContinueStatement *continueS);

	virtual void VisitOut(const Statement *statement);
	virtual void VisitOut(const DeclarationStatement *declaration);
	virtual void VisitOut(const AssignStatement *assignS);
	virtual void VisitOut(const ExpressionStatement *expressionS);
	virtual void VisitOut(const IfStatement *ifS);
	virtual void VisitOut(const WhileStatement *whileS);
	virtual void VisitOut(const RepeatStatement *repeatS);
	virtual void VisitOut(const BlockStatement *blockS);
	virtual void VisitOut(const ReturnStatement *returnS);
	virtual void VisitOut(const BreakStatement *breakS);
	virtual void VisitOut(const ContinueStatement *continueS);

	// Expressions

	virtual bool VisitIn(const Expression *expression);
	virtual bool VisitIn(const CallExpression *call);
	virtual bool VisitIn(const CastExpression *cast);
	virtual bool VisitIn(const Operand *operand);
	virtual bool VisitIn(const Literal *literal);
	virtual bool VisitIn(const Identifier *identifier);

	virtual void VisitOut(const Expression *expression);
	virtual void VisitOut(const CastExpression *cast);
	virtual void VisitOut(const CallExpression *call);
	virtual void VisitOut(const Operand *operand);
	virtual void VisitOut(const Literal *literal);
	virtual void VisitOut(const Identifier *identifier);

	// Literals

	virtual bool VisitIn(const VectorLiteral *literal);
	virtual bool VisitIn(const BooleanLiteral *literal);
	virtual bool VisitIn(const CharLiteral *literal);
	virtual bool VisitIn(const Int8Literal *literal);
	virtual bool VisitIn(const Int16Literal *literal);
	virtual bool VisitIn(const Int32Literal *literal);
	virtual bool VisitIn(const Int64Literal *literal);
	virtual bool VisitIn(const Float32Literal *literal);
	virtual bool VisitIn(const Float64Literal *literal);
	virtual bool VisitIn(const ComplexLiteral *literal);
	virtual bool VisitIn(const StringLiteral *literal);
	virtual bool VisitIn(const SymbolLiteral *literal);
	virtual bool VisitIn(const DatetimeLiteral *literal);
	virtual bool VisitIn(const MonthLiteral *literal);
	virtual bool VisitIn(const DateLiteral *literal);
	virtual bool VisitIn(const MinuteLiteral *literal);
	virtual bool VisitIn(const SecondLiteral *literal);
	virtual bool VisitIn(const TimeLiteral *literal);
	virtual bool VisitIn(const FunctionLiteral *literal);

	virtual void VisitOut(const VectorLiteral *literal);
	virtual void VisitOut(const BooleanLiteral *literal);
	virtual void VisitOut(const CharLiteral *literal);
	virtual void VisitOut(const Int8Literal *literal);
	virtual void VisitOut(const Int16Literal *literal);
	virtual void VisitOut(const Int32Literal *literal);
	virtual void VisitOut(const Int64Literal *literal);
	virtual void VisitOut(const Float32Literal *literal);
	virtual void VisitOut(const Float64Literal *literal);
	virtual void VisitOut(const ComplexLiteral *literal);
	virtual void VisitOut(const StringLiteral *literal);
	virtual void VisitOut(const SymbolLiteral *literal);
	virtual void VisitOut(const DatetimeLiteral *literal);
	virtual void VisitOut(const MonthLiteral *literal);
	virtual void VisitOut(const DateLiteral *literal);
	virtual void VisitOut(const MinuteLiteral *literal);
	virtual void VisitOut(const SecondLiteral *literal);
	virtual void VisitOut(const TimeLiteral *literal);
	virtual void VisitOut(const FunctionLiteral *literal);

	// Types

	virtual bool VisitIn(const Type *type);
	virtual bool VisitIn(const WildcardType *type);
	virtual bool VisitIn(const BasicType *type);
	virtual bool VisitIn(const FunctionType *type);
	virtual bool VisitIn(const ListType *type);
	virtual bool VisitIn(const DictionaryType *type);
	virtual bool VisitIn(const EnumerationType *type);
	virtual bool VisitIn(const TableType *type);
	virtual bool VisitIn(const KeyedTableType *type);

	virtual void VisitOut(const Type *type);
	virtual void VisitOut(const WildcardType *type);
	virtual void VisitOut(const BasicType *type);
	virtual void VisitOut(const FunctionType *type);
	virtual void VisitOut(const ListType *type);
	virtual void VisitOut(const DictionaryType *type);
	virtual void VisitOut(const EnumerationType *type);
	virtual void VisitOut(const TableType *type);
	virtual void VisitOut(const KeyedTableType *type);
};

}
