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
#include "HorseIR/Tree/Expressions/Literal.h"
#include "HorseIR/Tree/Expressions/Symbol.h"

#include "HorseIR/Tree/Types/Type.h"
#include "HorseIR/Tree/Types/ListType.h"
#include "HorseIR/Tree/Types/PrimitiveType.h"

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

void ConstVisitor::Visit(const Parameter *parameter)
{
	Visit(static_cast<const Node*>(parameter));
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

void ConstVisitor::Visit(const Literal<int64_t> *literal)
{
	Visit(static_cast<const Expression*>(literal));
}

void ConstVisitor::Visit(const Literal<double> *literal)
{
	Visit(static_cast<const Expression*>(literal));
}

void ConstVisitor::Visit(const Literal<std::string> *literal)
{
	Visit(static_cast<const Expression*>(literal));
}

void ConstVisitor::Visit(const Symbol *symbol)
{
	Visit(static_cast<const Expression*>(symbol));
}

void ConstVisitor::Visit(const Type *type)
{
	Visit(static_cast<const Node*>(type));
}

void ConstVisitor::Visit(const ListType *type)
{
	Visit(static_cast<const Type*>(type));
}

void ConstVisitor::Visit(const PrimitiveType *type)
{
	Visit(static_cast<const Type*>(type));
}

}
