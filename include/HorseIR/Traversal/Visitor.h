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
class Operand;
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
class DictionaryType;
class EnumerationType;
class FunctionType;
class KeyedTableType;
class ListType;
class TableType;

class Visitor
{
public:
	virtual void Visit(Node *node);

	virtual void Visit(Program *program);
	virtual void Visit(Module *module);
	virtual void Visit(ModuleContent *moduleContent);
	virtual void Visit(Import *import);
	virtual void Visit(MethodDeclaration *method);
	virtual void Visit(BuiltinMethod *method);
	virtual void Visit(Method *method);
	virtual void Visit(Parameter *parameter);
	virtual void Visit(Declaration *declaration);

	virtual void Visit(Statement *statement);
	virtual void Visit(AssignStatement *assign);
	virtual void Visit(ReturnStatement *ret);

	virtual void Visit(Expression *expression);
	virtual void Visit(Operand *operand);
	virtual void Visit(CallExpression *call);
	virtual void Visit(CastExpression *cast);
	virtual void Visit(Identifier *identifier);
	virtual void Visit(ModuleIdentifier *identifier);

	virtual void Visit(BoolLiteral *literal);
	virtual void Visit(Int8Literal *literal);
	virtual void Visit(Int16Literal *literal);
	virtual void Visit(Int32Literal *literal);
	virtual void Visit(Int64Literal *literal);
	virtual void Visit(Float32Literal *literal);
	virtual void Visit(Float64Literal *literal);
	virtual void Visit(StringLiteral *literal);
	virtual void Visit(SymbolLiteral *literal);
	virtual void Visit(DateLiteral *literal);
	virtual void Visit(FunctionLiteral *literal);

	virtual void Visit(Type *type);
	virtual void Visit(BasicType *type);
	virtual void Visit(DictionaryType *type);
	virtual void Visit(EnumerationType *type);
	virtual void Visit(FunctionType *type);
	virtual void Visit(KeyedTableType *type);
	virtual void Visit(ListType *type);
	virtual void Visit(TableType *type);
};

}
