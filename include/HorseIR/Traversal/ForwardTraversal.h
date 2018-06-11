#pragma once

#include "HorseIR/Traversal/Visitor.h"

#include "HorseIR/Tree/Program.h"
#include "HorseIR/Tree/Module.h"
#include "HorseIR/Tree/Method.h"
#include "HorseIR/Tree/Statements/AssignStatement.h"
#include "HorseIR/Tree/Expressions/CallExpression.h"

namespace HorseIR {

class ForwardTraversal : public Visitor
{
public:
	using Visitor::Visit;

	//TODO: these don't call Visit(Node *)
	void Visit(Program *program) override
	{
		for (auto module : program->GetModules())
		{
			module->Accept(*this);
		}
	}

	void Visit(Module *module) override
	{
		for (auto content : module->GetContents())
		{
			content->Accept(*this);
		}
	}

	void Visit(Method *method) override
	{
		method->GetReturnType()->Accept(*this);
		for (auto statement : method->GetStatements())
		{
			statement->Accept(*this);
		}
	}

	void Visit(AssignStatement *assign) override
	{
		assign->GetType()->Accept(*this);
		assign->GetExpression()->Accept(*this);
	}

	void Visit(ReturnStatement *ret) override
	{

	}

	void Visit(CallExpression *call) override
	{
		for (auto param : call->GetParameters())
		{
			param->Accept(*this);
		}
	}
};

}
