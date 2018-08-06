#include "HorseIR/Traversal/ConstForwardTraversal.h"

#include "HorseIR/Tree/Program.h"
#include "HorseIR/Tree/Module.h"
#include "HorseIR/Tree/Method.h"
#include "HorseIR/Tree/Statements/AssignStatement.h"
#include "HorseIR/Tree/Statements/ReturnStatement.h"
#include "HorseIR/Tree/Expressions/CallExpression.h"
#include "HorseIR/Tree/Expressions/CastExpression.h"

namespace HorseIR {

void ConstForwardTraversal::Visit(const Program *program)
{
	for (const auto& module : program->GetModules())
	{
		module->Accept(*this);
	}
}

void ConstForwardTraversal::Visit(const Module *module)
{
	for (const auto& content : module->GetContents())
	{
		content->Accept(*this);
	}
}

void ConstForwardTraversal::Visit(const Method *method)
{
	for (const auto& parameter : method->GetParameters())
	{
		parameter->Accept(*this);
	}
	const auto returnType = method->GetReturnType();
	if (returnType != nullptr)
	{
		returnType->Accept(*this);
	}
	for (const auto& statement : method->GetStatements())
	{
		statement->Accept(*this);
	}
}

void ConstForwardTraversal::Visit(const AssignStatement *assign)
{
	assign->GetType()->Accept(*this);
	assign->GetExpression()->Accept(*this);
}

void ConstForwardTraversal::Visit(const CallExpression *call)
{
	for (const auto& argument : call->GetArguments())
	{
		argument->Accept(*this);
	}
}

void ConstForwardTraversal::Visit(const CastExpression *cast)
{
	cast->GetExpression()->Accept(*this);
	cast->GetType()->Accept(*this);
}

void ConstForwardTraversal::Visit(const ReturnStatement *ret)
{
	ret->GetIdentifier()->Accept(*this);
}

}
