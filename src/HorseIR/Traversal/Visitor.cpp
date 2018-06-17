#include "HorseIR/Traversal/Visitor.h"

#include "HorseIR/Tree/Program.h"
#include "HorseIR/Tree/Module.h"
#include "HorseIR/Tree/ModuleContent.h"
#include "HorseIR/Tree/Import.h"
#include "HorseIR/Tree/Method.h"

#include "HorseIR/Tree/Statements/Statement.h"
#include "HorseIR/Tree/Statements/AssignStatement.h"
#include "HorseIR/Tree/Statements/ReturnStatement.h"

#include "HorseIR/Tree/Expressions/Expression.h"
#include "HorseIR/Tree/Expressions/CallExpression.h"
#include "HorseIR/Tree/Expressions/CastExpression.h"
#include "HorseIR/Tree/Expressions/Identifier.h"
#include "HorseIR/Tree/Expressions/Literal.h"
#include "HorseIR/Tree/Expressions/Symbol.h"

#include "HorseIR/Tree/Types/Type.h"
#include "HorseIR/Tree/Types/ListType.h"
#include "HorseIR/Tree/Types/PrimitiveType.h"
#include "HorseIR/Tree/Types/TableType.h"
#include "HorseIR/Tree/Types/WildcardType.h"

namespace HorseIR {

void Visitor::Visit(Node *node)
{

}

void Visitor::Visit(Program *program)
{
	Visit(static_cast<Node*>(program));
}

void Visitor::Visit(Module *module)
{

	Visit(static_cast<Node*>(module));
}

void Visitor::Visit(ModuleContent *moduleContent)
{
	Visit(static_cast<Node*>(moduleContent));
}

void Visitor::Visit(Import *import)
{
	Visit(static_cast<ModuleContent*>(import));
}

void Visitor::Visit(Method *method)
{
	Visit(static_cast<ModuleContent*>(method));
}

void Visitor::Visit(Statement *statement)
{
	Visit(static_cast<Node*>(statement));
}

void Visitor::Visit(AssignStatement *assign)
{
	Visit(static_cast<Statement*>(assign));
}

void Visitor::Visit(ReturnStatement *ret)
{
	Visit(static_cast<Statement*>(ret));
}

void Visitor::Visit(Expression *expression)
{
	Visit(static_cast<Node*>(expression));
}

void Visitor::Visit(CallExpression *call)
{
	Visit(static_cast<Expression*>(call));
}

void Visitor::Visit(CastExpression *cast)
{
	Visit(static_cast<Expression*>(cast));
}

void Visitor::Visit(Identifier *identifier)
{
	Visit(static_cast<Expression*>(identifier));
}

void Visitor::Visit(Literal<int64_t> *literal)
{
	Visit(static_cast<Expression*>(literal));
}

void Visitor::Visit(Literal<std::string> *literal)
{
	Visit(static_cast<Expression*>(literal));
}

void Visitor::Visit(Symbol *symbol)
{
	Visit(static_cast<Expression*>(symbol));
}

void Visitor::Visit(Type *type)
{
	Visit(static_cast<Node*>(type));
}

void Visitor::Visit(ListType *type)
{
	Visit(static_cast<Type*>(type));
}

void Visitor::Visit(PrimitiveType *type)
{
	Visit(static_cast<Type*>(type));
}

void Visitor::Visit(TableType *type)
{
	Visit(static_cast<Type*>(type));
}

void Visitor::Visit(WildcardType *type)
{
	Visit(static_cast<Type*>(type));
}

}
