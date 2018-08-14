#include "HorseIR/Traversal/Visitor.h"

#include "HorseIR/Tree/Program.h"
#include "HorseIR/Tree/Module.h"
#include "HorseIR/Tree/ModuleContent.h"
#include "HorseIR/Tree/Import.h"
#include "HorseIR/Tree/MethodDeclaration.h"
#include "HorseIR/Tree/BuiltinMethod.h"
#include "HorseIR/Tree/Method.h"

#include "HorseIR/Tree/Statements/Statement.h"
#include "HorseIR/Tree/Statements/AssignStatement.h"
#include "HorseIR/Tree/Statements/ReturnStatement.h"

#include "HorseIR/Tree/Expressions/Expression.h"
#include "HorseIR/Tree/Expressions/CallExpression.h"
#include "HorseIR/Tree/Expressions/CastExpression.h"
#include "HorseIR/Tree/Expressions/Identifier.h"
#include "HorseIR/Tree/Expressions/ModuleIdentifier.h"

#include "HorseIR/Tree/Expressions/Literals/BoolLiteral.h"
#include "HorseIR/Tree/Expressions/Literals/Int8Literal.h"
#include "HorseIR/Tree/Expressions/Literals/Int16Literal.h"
#include "HorseIR/Tree/Expressions/Literals/Int32Literal.h"
#include "HorseIR/Tree/Expressions/Literals/Int64Literal.h"
#include "HorseIR/Tree/Expressions/Literals/Float32Literal.h"
#include "HorseIR/Tree/Expressions/Literals/Float64Literal.h"
#include "HorseIR/Tree/Expressions/Literals/StringLiteral.h"
#include "HorseIR/Tree/Expressions/Literals/SymbolLiteral.h"
#include "HorseIR/Tree/Expressions/Literals/DateLiteral.h"
#include "HorseIR/Tree/Expressions/Literals/FunctionLiteral.h"

#include "HorseIR/Tree/Types/Type.h"
#include "HorseIR/Tree/Types/BasicType.h"
#include "HorseIR/Tree/Types/DictionaryType.h"
#include "HorseIR/Tree/Types/EnumerationType.h"
#include "HorseIR/Tree/Types/FunctionType.h"
#include "HorseIR/Tree/Types/KeyedTableType.h"
#include "HorseIR/Tree/Types/ListType.h"
#include "HorseIR/Tree/Types/TableType.h"

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

void Visitor::Visit(MethodDeclaration *method)
{
	Visit(static_cast<ModuleContent*>(method));
}

void Visitor::Visit(BuiltinMethod *method)
{
	Visit(static_cast<MethodDeclaration*>(method));
}

void Visitor::Visit(Method *method)
{
	Visit(static_cast<MethodDeclaration*>(method));
}

void Visitor::Visit(Declaration *declaration)
{
	Visit(static_cast<Node*>(declaration));
}

void Visitor::Visit(Parameter *parameter)
{
	Visit(static_cast<Declaration*>(parameter));
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

void Visitor::Visit(ModuleIdentifier *identifier)
{
	Visit(static_cast<Expression*>(identifier));
}

void Visitor::Visit(BoolLiteral *literal)
{
	Visit(static_cast<Expression*>(literal));
}

void Visitor::Visit(Int8Literal *literal)
{
	Visit(static_cast<Expression*>(literal));
}

void Visitor::Visit(Int16Literal *literal)
{
	Visit(static_cast<Expression*>(literal));
}

void Visitor::Visit(Int32Literal *literal)
{
	Visit(static_cast<Expression*>(literal));
}

void Visitor::Visit(Int64Literal *literal)
{
	Visit(static_cast<Expression*>(literal));
}

void Visitor::Visit(Float32Literal *literal)
{
	Visit(static_cast<Expression*>(literal));
}

void Visitor::Visit(Float64Literal *literal)
{
	Visit(static_cast<Expression*>(literal));
}

void Visitor::Visit(StringLiteral *literal)
{
	Visit(static_cast<Expression*>(literal));
}

void Visitor::Visit(SymbolLiteral *literal)
{
	Visit(static_cast<Expression*>(literal));
}

void Visitor::Visit(DateLiteral *literal)
{
	Visit(static_cast<Expression*>(literal));
}

void Visitor::Visit(FunctionLiteral *literal)
{
	Visit(static_cast<Expression*>(literal));
}

void Visitor::Visit(Type *type)
{
	Visit(static_cast<Node*>(type));
}

void Visitor::Visit(BasicType *type)
{
	Visit(static_cast<Type*>(type));
}

void Visitor::Visit(DictionaryType *type)
{
	Visit(static_cast<Type*>(type));
}

void Visitor::Visit(EnumerationType *type)
{
	Visit(static_cast<Type*>(type));
}

void Visitor::Visit(FunctionType *type)
{
	Visit(static_cast<Type*>(type));
}

void Visitor::Visit(KeyedTableType *type)
{
	Visit(static_cast<Type*>(type));
}

void Visitor::Visit(ListType *type)
{
	Visit(static_cast<Type*>(type));
}

void Visitor::Visit(TableType *type)
{
	Visit(static_cast<Type*>(type));
}

}
