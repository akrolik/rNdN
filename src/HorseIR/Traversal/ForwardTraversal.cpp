#include "HorseIR/Traversal/ForwardTraversal.h"

#include "HorseIR/Tree/Program.h"
#include "HorseIR/Tree/Module.h"
#include "HorseIR/Tree/Import.h"
#include "HorseIR/Tree/Method.h"
#include "HorseIR/Tree/Declaration.h"
#include "HorseIR/Tree/Statements/AssignStatement.h"
#include "HorseIR/Tree/Statements/ReturnStatement.h"
#include "HorseIR/Tree/Expressions/CallExpression.h"
#include "HorseIR/Tree/Expressions/CastExpression.h"
#include "HorseIR/Tree/Expressions/Literals/FunctionLiteral.h"
#include "HorseIR/Tree/Types/DictionaryType.h"
#include "HorseIR/Tree/Types/ListType.h"

namespace HorseIR {

void ForwardTraversal::Visit(Program *program)
{
	for (auto& module : program->GetModules())
	{
		module->Accept(*this);
	}
	Visitor::Visit(program);
}

void ForwardTraversal::Visit(Module *module)
{
	for (auto& content : module->GetContents())
	{
		content->Accept(*this);
	}
	Visitor::Visit(module);
}

void ForwardTraversal::Visit(Import *import)
{
	import->GetIdentifier()->Accept(*this);
	Visitor::Visit(import);
}

void ForwardTraversal::Visit(Method *method)
{
	for (auto& parameter : method->GetParameters())
	{
		parameter->Accept(*this);
	}
	auto returnType = method->GetReturnType();
	if (returnType != nullptr)
	{
		returnType->Accept(*this);
	}
	for (auto& statement : method->GetStatements())
	{
		statement->Accept(*this);
	}
	Visitor::Visit(method);
}

void ForwardTraversal::Visit(Declaration *declaration)
{
	declaration->GetType()->Accept(*this);
	Visitor::Visit(declaration);
}

void ForwardTraversal::Visit(AssignStatement *assign)
{
	assign->GetDeclaration()->Accept(*this);
	assign->GetExpression()->Accept(*this);
	Visitor::Visit(assign);
}

void ForwardTraversal::Visit(ReturnStatement *ret)
{
	ret->GetIdentifier()->Accept(*this);
	Visitor::Visit(ret);
}

void ForwardTraversal::Visit(CallExpression *call)
{
	for (auto& argument : call->GetArguments())
	{
		argument->Accept(*this);
	}
	Visitor::Visit(call);
}

void ForwardTraversal::Visit(CastExpression *cast)
{
	cast->GetExpression()->Accept(*this);
	cast->GetCastType()->Accept(*this);
	Visitor::Visit(cast);
}

void ForwardTraversal::Visit(FunctionLiteral *literal)
{
	literal->GetIdentifier()->Accept(*this);
	Visitor::Visit(literal);
}

void ForwardTraversal::Visit(DictionaryType *type)
{
	type->GetKeyType()->Accept(*this);
	type->GetValueType()->Accept(*this);
	Visitor::Visit(type);
}

void ForwardTraversal::Visit(ListType *type)
{
	type->GetElementType()->Accept(*this);
	Visitor::Visit(type);
}

}
