#pragma once

#include <cstdint>
#include <string>

namespace HorseIR {

class Node;

class Program;
class Module;
class ModuleContent;
class Import;
class Method;
class Parameter;

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
class ListType;
class PrimitiveType;

class Visitor
{
public:
	virtual void Visit(Node *node);

	virtual void Visit(Program *program);
	virtual void Visit(Module *module);
	virtual void Visit(ModuleContent *moduleContent);
	virtual void Visit(Import *import);
	virtual void Visit(Method *method);
	virtual void Visit(Parameter *parameter);

	virtual void Visit(Statement *statement);
	virtual void Visit(AssignStatement *assign);
	virtual void Visit(ReturnStatement *ret);

	virtual void Visit(Expression *expression);
	virtual void Visit(CallExpression *call);
	virtual void Visit(CastExpression *cast);
	virtual void Visit(Identifier *identifier);
	virtual void Visit(Literal<std::string> *literal);
	virtual void Visit(Literal<int64_t> *literal);
	virtual void Visit(Symbol *symbol);

	virtual void Visit(Type *type);
	virtual void Visit(ListType *type);
	virtual void Visit(PrimitiveType *type);
};

}
