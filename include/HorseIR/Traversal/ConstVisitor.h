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

template<class T>
class Literal;
class Symbol;

class Type;
class BasicType;
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
	virtual void Visit(const Literal<int64_t> *literal);
	virtual void Visit(const Literal<double> *literal);
	virtual void Visit(const Literal<std::string> *literal);
	virtual void Visit(const Symbol *symbol);

	virtual void Visit(const Type *type);
	virtual void Visit(const BasicType *type);
	virtual void Visit(const ListType *type);
	virtual void Visit(const TableType *type);
};

}
