#include "HorseIR/Analysis/Shape/ShapeAnalysisDumper.h"

#include "HorseIR/Analysis/Shape/ShapeUtils.h"
#include "HorseIR/Tree/BuiltinMethod.h"
#include "HorseIR/Tree/Method.h"
#include "HorseIR/Tree/Parameter.h"
#include "HorseIR/Tree/Expressions/CallExpression.h"
#include "HorseIR/Tree/Expressions/CastExpression.h"
#include "HorseIR/Tree/Expressions/Expression.h"
#include "HorseIR/Tree/Expressions/Identifier.h"
#include "HorseIR/Tree/Expressions/Literals/BoolLiteral.h"
#include "HorseIR/Tree/Expressions/Literals/DateLiteral.h"
#include "HorseIR/Tree/Expressions/Literals/Int8Literal.h"
#include "HorseIR/Tree/Expressions/Literals/Int16Literal.h"
#include "HorseIR/Tree/Expressions/Literals/Int32Literal.h"
#include "HorseIR/Tree/Expressions/Literals/Int64Literal.h"
#include "HorseIR/Tree/Expressions/Literals/Float32Literal.h"
#include "HorseIR/Tree/Expressions/Literals/Float64Literal.h"
#include "HorseIR/Tree/Expressions/Literals/FunctionLiteral.h"
#include "HorseIR/Tree/Expressions/Literals/StringLiteral.h"
#include "HorseIR/Tree/Expressions/Literals/SymbolLiteral.h"
#include "HorseIR/Tree/Statements/AssignStatement.h"
#include "HorseIR/Tree/Statements/ReturnStatement.h"

#include "Utils/Logger.h"

namespace HorseIR {

std::string Indentation(unsigned int indentation)
{
	return std::string(indentation, '\t');
}

void ShapeAnalysisDumper::Dump(const MethodDeclaration *method, const ShapeResults *shapes)
{
	m_results = shapes;

	OpenContext(nullptr);
	method->Accept(*this);
	CloseContext();

	m_results = nullptr;
}

void ShapeAnalysisDumper::Visit(const Method *method)
{
	std::string code = std::string((method->IsKernel()) ? "kernel" : "def") + " " + method->GetName() + "(";

	bool first = true;
	for (const auto& parameter : method->GetParameters())
	{
		if (!first)
		{
			code += ", ";
		}
		first = false;

		auto name = parameter->GetName();
		auto shape = m_shapes.top()->GetShape(name);
		code += name + ":" + shape->ToString();
	}
	code += ")";
	if (method->GetReturnType() != nullptr)
	{
		code += " : " + m_shapes.top()->GetReturnShape()->ToString();
	}

	std::string indentation = Indentation(m_indentation);
	Utils::Logger::LogInfo(indentation + code, 0, true, Utils::Logger::NoPrefix);
	Utils::Logger::LogInfo(indentation + "{", 0, true, Utils::Logger::NoPrefix);

	m_indentation++;
	for (const auto& statement : method->GetStatements())
	{
		statement->Accept(*this);
	}
	m_indentation--;

	Utils::Logger::LogInfo(indentation + "}", 0, true, Utils::Logger::NoPrefix);
}

void ShapeAnalysisDumper::Visit(const AssignStatement *assign)
{
	auto declaration = assign->GetDeclaration();
	auto expression = assign->GetExpression();

	// Print assignment statement in the form

	Utils::Logger::LogInfo(Indentation(m_indentation), 0, false, Utils::Logger::NoPrefix);
	declaration->Accept(*this);
	Utils::Logger::LogInfo(" = ", 0, false, Utils::Logger::NoPrefix);
	expression->Accept(*this);
	Utils::Logger::LogBlank(Utils::Logger::NoPrefix);
}

void ShapeAnalysisDumper::Visit(const ReturnStatement *ret)
{
	auto name = ret->GetIdentifier()->GetString();
	std::string code = Indentation(m_indentation) + "return " + name + ":" + m_shapes.top()->GetShape(name)->ToString();
	Utils::Logger::LogInfo(code, 0, true, Utils::Logger::NoPrefix);
}

void ShapeAnalysisDumper::Visit(const Declaration *declaration)
{
	auto name = declaration->GetName();
	auto shape = m_shapes.top()->GetShape(name);
	Utils::Logger::LogInfo(name + ":" + shape->ToString(), 0, false, Utils::Logger::NoPrefix);
}

void ShapeAnalysisDumper::Visit(const Expression *expression)
{
	auto shape = m_shapes.top()->GetShape(expression);
	Utils::Logger::LogInfo(shape->ToString(), 0, false, Utils::Logger::NoPrefix);
}

void ShapeAnalysisDumper::Visit(const CallExpression *call)
{
	auto method = call->GetMethod();
	auto definition = (method->GetKind() == MethodDeclaration::Kind::Definition);

	std::string code = "@" + method->GetName() + "(";
	bool first = true;
	for (const auto& argument : call->GetArguments())
	{
		if (!first)
		{
			code += ", ";
		}
		first = false;

		auto shape = m_shapes.top()->GetShape(argument);
		code += argument->ToString() + ":" + shape->ToString();
	}
	code += ")";
	Utils::Logger::LogInfo(code, 0, definition, Utils::Logger::NoPrefix);
	
	if (definition)
	{
		OpenContext(call);
		method->Accept(*this);
		CloseContext();
	}
}

void ShapeAnalysisDumper::Visit(const CastExpression *cast)
{
	// Casting does not change the shape of the expression, so we only print the contents

	cast->GetExpression()->Accept(*this);
}

void ShapeAnalysisDumper::OpenContext(const CallExpression *call)
{
	if (call != nullptr)
	{
		m_indentation++;
	}
	m_shapes.push(m_results->GetInvocationShapes(call));
}

void ShapeAnalysisDumper::CloseContext()
{
	m_shapes.pop();
	if (m_indentation > 0)
	{
		m_indentation--;
	}
}

}
