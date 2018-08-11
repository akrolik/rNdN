#pragma once

#include <cstdint>
#include <string>

namespace HorseIR {

class Node;

class Program;
class Module;
class ModuleContent;
class Import;
class MethodDeclaration;
class BuiltinMethod;
class Method;
class Parameter;
class Declaration;

class Statement;
class AssignStatement;
class ReturnStatement;

class Expression;
class CallExpression;
class CastExpression;
class Identifier;
class ModuleIdentifier;

class BoolLiteral;
class Int8Literal;
class Int16Literal;
class Int32Literal;
class Int64Literal;
class Float32Literal;
class Float64Literal;
class StringLiteral;
class SymbolLiteral;
class DateLiteral;
class FunctionLiteral;

class Type;
class BasicType;
class FunctionType;
class DictionaryType;
class ListType;
class TableType;

class ConstVisitor
{
public:
	virtual void Visit(const Node *node);

	virtual void Visit(const Program *program);
	virtual void Visit(const Module *module);
	virtual void Visit(const ModuleContent *moduleContent);
	virtual void Visit(const Import *import);
	virtual void Visit(const MethodDeclaration *method);
	virtual void Visit(const BuiltinMethod *method);
	virtual void Visit(const Method *method);
	virtual void Visit(const Parameter *parameter);
	virtual void Visit(const Declaration *declaration);

	virtual void Visit(const Statement *statement);
	virtual void Visit(const AssignStatement *assign);
	virtual void Visit(const ReturnStatement *ret);

	virtual void Visit(const Expression *expression);
	virtual void Visit(const CallExpression *call);
	virtual void Visit(const CastExpression *cast);
	virtual void Visit(const Identifier *identifier);
	virtual void Visit(const ModuleIdentifier *identifier);

	virtual void Visit(const BoolLiteral *literal);
	virtual void Visit(const Int8Literal *literal);
	virtual void Visit(const Int16Literal *literal);
	virtual void Visit(const Int32Literal *literal);
	virtual void Visit(const Int64Literal *literal);
	virtual void Visit(const Float32Literal *literal);
	virtual void Visit(const Float64Literal *literal);
	virtual void Visit(const StringLiteral *literal);
	virtual void Visit(const SymbolLiteral *literal);
	virtual void Visit(const DateLiteral *literal);
	virtual void Visit(const FunctionLiteral *literal);

	virtual void Visit(const Type *type);
	virtual void Visit(const BasicType *type);
	virtual void Visit(const FunctionType *type);
	virtual void Visit(const DictionaryType *type);
	virtual void Visit(const ListType *type);
	virtual void Visit(const TableType *type);
};

}
