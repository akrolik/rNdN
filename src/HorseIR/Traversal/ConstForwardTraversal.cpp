#include "HorseIR/Traversal/ConstForwardTraversal.h"

#include "HorseIR/Tree/Program.h"
#include "HorseIR/Tree/Module.h"
#include "HorseIR/Tree/Method.h"
#include "HorseIR/Tree/Declaration.h"
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
	ConstVisitor::Visit(program);
}

void ConstForwardTraversal::Visit(const Module *module)
{
	for (const auto& content : module->GetContents())
	{
		content->Accept(*this);
	}
	ConstVisitor::Visit(module);
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
	ConstVisitor::Visit(method);
}

void ConstForwardTraversal::Visit(const AssignStatement *assign)
{
	assign->GetDeclaration()->Accept(*this);
	assign->GetExpression()->Accept(*this);
	ConstVisitor::Visit(assign);
}

void ConstForwardTraversal::Visit(const Declaration *declaration)
{
	declaration->GetType()->Accept(*this);
	ConstVisitor::Visit(declaration);
}

void ConstForwardTraversal::Visit(const CallExpression *call)
{
	for (const auto& argument : call->GetArguments())
	{
		argument->Accept(*this);
	}
	ConstVisitor::Visit(call);
}

void ConstForwardTraversal::Visit(const CastExpression *cast)
{
	cast->GetExpression()->Accept(*this);
	cast->GetCastType()->Accept(*this);
	ConstVisitor::Visit(cast);
}

void ConstForwardTraversal::Visit(const ReturnStatement *ret)
{
	ret->GetIdentifier()->Accept(*this);
	ConstVisitor::Visit(ret);
}

}
