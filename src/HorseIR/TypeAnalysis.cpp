#include "HorseIR/TypeAnalysis.h"

#include "HorseIR/Tree/BuiltinMethod.h"
#include "HorseIR/Tree/Method.h"
#include "HorseIR/Tree/Expressions/CallExpression.h"
#include "HorseIR/Tree/Expressions/CastExpression.h"
#include "HorseIR/Tree/Expressions/Expression.h"
#include "HorseIR/Tree/Expressions/Identifier.h"
#include "HorseIR/Tree/Expressions/Literal.h"
#include "HorseIR/Tree/Expressions/Symbol.h"
#include "HorseIR/Tree/Statements/AssignStatement.h"
#include "HorseIR/Tree/Types/ListType.h"
#include "HorseIR/Tree/Types/TableType.h"

#include "Utils/Logger.h"

namespace HorseIR {

void TypeAnalysis::Analyze(HorseIR::Method *method)
{
	method->Accept(*this);
}

void TypeAnalysis::Visit(AssignStatement *assign)
{
	// Traverse all children of the assignment

	ForwardTraversal::Visit(assign);

	// Update the declaration with the shape propagated from the expression

	auto declarationType = assign->GetDeclaration()->GetType();
	auto expressionType = assign->GetExpression()->GetType();

	if (*declarationType != *expressionType)
	{
		// Utils::Logger::LogError("Expression type '" + expressionType->ToString() + "' does not match destination type '" + declarationType->ToString() + "'");
	}
}

void TypeAnalysis::Visit(CallExpression *call)
{
	// Collect type information for the arguments

	ForwardTraversal::Visit(call);

	// Analyze the function according to the type rules

	auto method = call->GetMethod();
	switch (method->GetKind())
	{
		case HorseIR::MethodDeclaration::Kind::Builtin:
			call->SetType(AnalyzeCall(static_cast<const HorseIR::BuiltinMethod *>(method), call->GetArguments()));
			break;
		case HorseIR::MethodDeclaration::Kind::Definition:
			call->SetType(static_cast<const HorseIR::Method *>(method)->GetReturnType());
			break;
	}
}

const Type *TypeAnalysis::AnalyzeCall(const BuiltinMethod *method, const std::vector<Expression *>& arguments)
{
	switch (method->GetKind())
	{
		case BuiltinMethod::Kind::Absolute:
		case BuiltinMethod::Kind::Negate:
		case BuiltinMethod::Kind::Ceiling:
		case BuiltinMethod::Kind::Floor:
		case BuiltinMethod::Kind::Round:
		case BuiltinMethod::Kind::Reciprocal:
		case BuiltinMethod::Kind::Sign:
		case BuiltinMethod::Kind::Pi:
			return nullptr;
		case BuiltinMethod::Kind::Not:
			return new BasicType(BasicType::Kind::Bool);
		case BuiltinMethod::Kind::Logarithm:
		case BuiltinMethod::Kind::Logarithm2:
		case BuiltinMethod::Kind::Logarithm10:
		case BuiltinMethod::Kind::SquareRoot:
		case BuiltinMethod::Kind::Exponential:
		case BuiltinMethod::Kind::Cosine:
		case BuiltinMethod::Kind::Sine:
		case BuiltinMethod::Kind::Tangent:
		case BuiltinMethod::Kind::InverseCosine:
		case BuiltinMethod::Kind::InverseSine:
		case BuiltinMethod::Kind::InverseTangent:
		case BuiltinMethod::Kind::HyperbolicCosine:
		case BuiltinMethod::Kind::HyperbolicSine:
		case BuiltinMethod::Kind::HyperbolicTangent:
		case BuiltinMethod::Kind::HyperbolicInverseCosine:
		case BuiltinMethod::Kind::HyperbolicInverseSine:
		case BuiltinMethod::Kind::HyperbolicInverseTangent:
			return nullptr;
		case BuiltinMethod::Kind::Less:
		case BuiltinMethod::Kind::Greater:
		case BuiltinMethod::Kind::LessEqual:
		case BuiltinMethod::Kind::GreaterEqual:
		case BuiltinMethod::Kind::Equal:
		case BuiltinMethod::Kind::NotEqual:
			return new BasicType(BasicType::Kind::Bool);
		case BuiltinMethod::Kind::Plus:
		case BuiltinMethod::Kind::Minus:
			return nullptr;
		case BuiltinMethod::Kind::Multiply:
			return new BasicType(BasicType::Kind::Float32);
		case BuiltinMethod::Kind::Divide:
		case BuiltinMethod::Kind::Power:
		case BuiltinMethod::Kind::LogarithmBase:
		case BuiltinMethod::Kind::Modulo:
			return nullptr;
		case BuiltinMethod::Kind::And:
		case BuiltinMethod::Kind::Or:
		case BuiltinMethod::Kind::Nand:
		case BuiltinMethod::Kind::Nor:
		case BuiltinMethod::Kind::Xor:
			return nullptr;
		case BuiltinMethod::Kind::Compress:
			return arguments.at(1)->GetType();
		case BuiltinMethod::Kind::Count:
		case BuiltinMethod::Kind::Sum:
		case BuiltinMethod::Kind::Average:
		case BuiltinMethod::Kind::Minimum:
		case BuiltinMethod::Kind::Maximum:
			return new BasicType(BasicType::Kind::Float32);
		case BuiltinMethod::Kind::Enlist:
			return new ListType(arguments.at(0)->GetType());
		case BuiltinMethod::Kind::Table:
			return new TableType();
		case BuiltinMethod::Kind::ColumnValue:
			return nullptr;
		case BuiltinMethod::Kind::LoadTable:
			return new TableType();
		case BuiltinMethod::Kind::Fill:
			return nullptr;
		default:
			Utils::Logger::LogError("Type analysis does not support builtin method " + method->GetName());
	}
}

void TypeAnalysis::Visit(CastExpression *cast)
{
	cast->SetType(cast->GetCastType());
}

void TypeAnalysis::Visit(Identifier *identifier)
{
	identifier->SetType(identifier->GetDeclaration()->GetType());
}

void TypeAnalysis::Visit(Literal<int64_t> *literal)
{
	literal->SetType(literal->GetLiteralType());
}

void TypeAnalysis::Visit(Literal<double> *literal)
{
	literal->SetType(literal->GetLiteralType());
}

void TypeAnalysis::Visit(Literal<std::string> *literal)
{
	literal->SetType(literal->GetLiteralType());
}

void TypeAnalysis::Visit(Symbol *symbol)
{
	symbol->SetType(symbol->GetLiteralType());
}

}

