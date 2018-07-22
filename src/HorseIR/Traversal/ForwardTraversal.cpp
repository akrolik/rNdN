#include "HorseIR/Traversal/ForwardTraversal.h"

#include "HorseIR/Tree/Program.h"
#include "HorseIR/Tree/Module.h"
#include "HorseIR/Tree/Method.h"
#include "HorseIR/Tree/Statements/AssignStatement.h"
#include "HorseIR/Tree/Statements/ReturnStatement.h"
#include "HorseIR/Tree/Expressions/CallExpression.h"

namespace HorseIR {

void ForwardTraversal::Visit(Program *program)
{
	for (auto& module : program->GetModules())
	{
		module->Accept(*this);
	}
}

void ForwardTraversal::Visit(Module *module)
{
	for (auto& content : module->GetContents())
	{
		content->Accept(*this);
	}
}

void ForwardTraversal::Visit(Method *method)
{
	for (auto& parameter : method->GetParameters())
	{
		parameter->Accept(*this);
	}
	method->GetReturnType()->Accept(*this);
	for (auto& statement : method->GetStatements())
	{
		statement->Accept(*this);
	}
}

void ForwardTraversal::Visit(AssignStatement *assign)
{
	assign->GetType()->Accept(*this);
	assign->GetExpression()->Accept(*this);
}

void ForwardTraversal::Visit(CallExpression *call)
{
	for (auto& argument : call->GetArguments())
	{
		argument->Accept(*this);
	}
}

void ForwardTraversal::Visit(ReturnStatement *ret)
{
	ret->GetIdentifier()->Accept(*this);
}

}
