#include "HorseIR/Traversal/ConstVisitor.h"

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
#include "HorseIR/Tree/Types/FunctionType.h"
#include "HorseIR/Tree/Types/DictionaryType.h"
#include "HorseIR/Tree/Types/ListType.h"
#include "HorseIR/Tree/Types/TableType.h"

namespace HorseIR {

void ConstVisitor::Visit(const Node *node)
{

}

void ConstVisitor::Visit(const Program *program)
{
	Visit(static_cast<const Node*>(program));
}

void ConstVisitor::Visit(const Module *module)
{
	Visit(static_cast<const Node*>(module));
}

void ConstVisitor::Visit(const ModuleContent *moduleContent)
{
	Visit(static_cast<const Node*>(moduleContent));
}

void ConstVisitor::Visit(const Import *import)
{
	Visit(static_cast<const ModuleContent*>(import));
}

void ConstVisitor::Visit(const MethodDeclaration *method)
{
	Visit(static_cast<const ModuleContent*>(method));
}

void ConstVisitor::Visit(const BuiltinMethod *method)
{
	Visit(static_cast<const MethodDeclaration*>(method));
}

void ConstVisitor::Visit(const Method *method)
{
	Visit(static_cast<const MethodDeclaration*>(method));
}

void ConstVisitor::Visit(const Declaration *declaration)
{
	Visit(static_cast<const Node*>(declaration));
}

void ConstVisitor::Visit(const Parameter *parameter)
{
	Visit(static_cast<const Declaration*>(parameter));
}

void ConstVisitor::Visit(const Statement *statement)
{
	Visit(static_cast<const Node*>(statement));
}

void ConstVisitor::Visit(const AssignStatement *assign)
{
	Visit(static_cast<const Statement*>(assign));
}

void ConstVisitor::Visit(const ReturnStatement *ret)
{
	Visit(static_cast<const Statement*>(ret));
}

void ConstVisitor::Visit(const Expression *expression)
{
	Visit(static_cast<const Node*>(expression));
}

void ConstVisitor::Visit(const CallExpression *call)
{
	Visit(static_cast<const Expression*>(call));
}

void ConstVisitor::Visit(const CastExpression *cast)
{
	Visit(static_cast<const Expression*>(cast));
}

void ConstVisitor::Visit(const Identifier *identifier)
{
	Visit(static_cast<const Expression*>(identifier));
}

void ConstVisitor::Visit(const ModuleIdentifier *identifier)
{
	Visit(static_cast<const Expression*>(identifier));
}

void ConstVisitor::Visit(const BoolLiteral *literal)
{
	Visit(static_cast<const Expression*>(literal));
}

void ConstVisitor::Visit(const Int8Literal *literal)
{
	Visit(static_cast<const Expression*>(literal));
}

void ConstVisitor::Visit(const Int16Literal *literal)
{
	Visit(static_cast<const Expression*>(literal));
}

void ConstVisitor::Visit(const Int32Literal *literal)
{
	Visit(static_cast<const Expression*>(literal));
}

void ConstVisitor::Visit(const Int64Literal *literal)
{
	Visit(static_cast<const Expression*>(literal));
}

void ConstVisitor::Visit(const Float32Literal *literal)
{
	Visit(static_cast<const Expression*>(literal));
}

void ConstVisitor::Visit(const Float64Literal *literal)
{
	Visit(static_cast<const Expression*>(literal));
}

void ConstVisitor::Visit(const StringLiteral *literal)
{
	Visit(static_cast<const Expression*>(literal));
}

void ConstVisitor::Visit(const SymbolLiteral *literal)
{
	Visit(static_cast<const Expression*>(literal));
}

void ConstVisitor::Visit(const DateLiteral *literal)
{
	Visit(static_cast<const Expression*>(literal));
}

void ConstVisitor::Visit(const FunctionLiteral *literal)
{
	Visit(static_cast<const Expression*>(literal));
}

void ConstVisitor::Visit(const Type *type)
{
	Visit(static_cast<const Node*>(type));
}

void ConstVisitor::Visit(const BasicType *type)
{
	Visit(static_cast<const Type*>(type));
}

void ConstVisitor::Visit(const FunctionType *type)
{
	Visit(static_cast<const BasicType*>(type));
}

void ConstVisitor::Visit(const DictionaryType *type)
{
	Visit(static_cast<const Type*>(type));
}

void ConstVisitor::Visit(const ListType *type)
{
	Visit(static_cast<const Type*>(type));
}

void ConstVisitor::Visit(const TableType *type)
{
	Visit(static_cast<const Type*>(type));
}

}
