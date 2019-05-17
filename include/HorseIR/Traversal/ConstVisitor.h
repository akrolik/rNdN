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
class Declaration;
class Parameter;

class Statement;
class LabelledStatement;
class AssignStatement;
class IfStatement;
class WhileStatement;
class RepeatStatement;
class GotoStatement;
class SwitchStatement;
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
class BasicType;
class FunctionType;
class ListType;
class DictionaryType;
class EnumerationType;
class TableType;
class KeyedTableType;

class ConstVisitor
{
public:
	// Node superclass

	virtual void Visit(const Node *node);

	// Modules

	virtual void Visit(const Program *program);
	virtual void Visit(const Module *module);
	virtual void Visit(const ModuleContent *moduleContent);
	virtual void Visit(const ImportDirective *import);
	virtual void Visit(const GlobalDeclaration *global);
	virtual void Visit(const FunctionDeclaration *function);
	virtual void Visit(const BuiltinFunction *function);
	virtual void Visit(const Function *function);
	virtual void Visit(const Declaration *declaration);
	virtual void Visit(const Parameter *parameter);

	// Statements

	virtual void Visit(const Statement *statement);
	virtual void Visit(const LabelledStatement *labelledS);
	virtual void Visit(const AssignStatement *assignS);
        virtual void Visit(const IfStatement *ifS);
	virtual void Visit(const WhileStatement *whileS);
	virtual void Visit(const RepeatStatement *repeatS);
	virtual void Visit(const GotoStatement *gotoS);
	virtual void Visit(const SwitchStatement *switchS);
	virtual void Visit(const ReturnStatement *returnS);
	virtual void Visit(const BreakStatement *breakS);
	virtual void Visit(const ContinueStatement *continueS);

	// Expressions

	virtual void Visit(const Expression *expression);
	virtual void Visit(const CallExpression *call);
	virtual void Visit(const CastExpression *cast);
	virtual void Visit(const Operand *operand);
	virtual void Visit(const Literal *literal);
	virtual void Visit(const Identifier *identifier);

	// Literals

	virtual void Visit(const VectorLiteral *literal);
	virtual void Visit(const BooleanLiteral *literal);
	virtual void Visit(const CharLiteral *literal);
	virtual void Visit(const Int8Literal *literal);
	virtual void Visit(const Int16Literal *literal);
	virtual void Visit(const Int32Literal *literal);
	virtual void Visit(const Int64Literal *literal);
	virtual void Visit(const Float32Literal *literal);
	virtual void Visit(const Float64Literal *literal);
	virtual void Visit(const ComplexLiteral *literal);
	virtual void Visit(const StringLiteral *literal);
	virtual void Visit(const SymbolLiteral *literal);
	virtual void Visit(const DatetimeLiteral *literal);
	virtual void Visit(const MonthLiteral *literal);
	virtual void Visit(const DateLiteral *literal);
	virtual void Visit(const MinuteLiteral *literal);
	virtual void Visit(const SecondLiteral *literal);
	virtual void Visit(const TimeLiteral *literal);
	virtual void Visit(const FunctionLiteral *literal);

	// Types

	virtual void Visit(const Type *type);
	virtual void Visit(const BasicType *type);
	virtual void Visit(const FunctionType *type);
	virtual void Visit(const ListType *type);
	virtual void Visit(const DictionaryType *type);
	virtual void Visit(const EnumerationType *type);
	virtual void Visit(const TableType *type);
	virtual void Visit(const KeyedTableType *type);
};

}
