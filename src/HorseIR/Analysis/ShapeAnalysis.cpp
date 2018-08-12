#include "HorseIR/Analysis/ShapeAnalysis.h"

#include "HorseIR/Tree/BuiltinMethod.h"
#include "HorseIR/Tree/Method.h"
#include "HorseIR/Tree/Parameter.h"
#include "HorseIR/Tree/Expressions/CallExpression.h"
#include "HorseIR/Tree/Expressions/CastExpression.h"
#include "HorseIR/Tree/Expressions/Expression.h"
#include "HorseIR/Tree/Expressions/Identifier.h"
#include "HorseIR/Tree/Expressions/Literals/Literal.h"
#include "HorseIR/Tree/Statements/AssignStatement.h"

#include "Utils/Logger.h"

namespace HorseIR {

void ShapeAnalysis::Analyze(HorseIR::Method *method)
{
	method->Accept(*this);
}

void ShapeAnalysis::Visit(Parameter *parameter)
{
	// Check to make sure the correct input shapes have been set

	// if (parameter->GetShape() == nullptr)
	// {
	// 	Utils::Logger::LogError("Shape analysis missing input shape for parameter '" + parameter->GetName() + "'");
	// }
}

void ShapeAnalysis::Visit(AssignStatement *assign)
{
	// Traverse all children of the assignment

	ForwardTraversal::Visit(assign);

	// Update the declaration with the shape propagated from the expression

	auto declaration = assign->GetDeclaration();
	auto expression = assign->GetExpression();

	// declaration->SetShape(expression->GetShape());
}

void ShapeAnalysis::Visit(CallExpression *call)
{
	// Collect shape information for the arguments

	ForwardTraversal::Visit(call);

	// Analyze the function according to the shape rules

	auto method = call->GetMethod();
	switch (method->GetKind())
	{
		case HorseIR::MethodDeclaration::Kind::Builtin:
			// call->SetShape(AnalyzeCall(static_cast<const HorseIR::BuiltinMethod *>(method), call->GetArguments()));
			break;
		case HorseIR::MethodDeclaration::Kind::Definition:
			Utils::Logger::LogError("Shape analysis for user defined functions not implemented");
			break;
	}
}

Shape *ShapeAnalysis::AnalyzeCall(const BuiltinMethod *method, const std::vector<Expression *>& arguments)
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
		case BuiltinMethod::Kind::Not:
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
		{
			// const auto argumentShape = arguments.at(0)->GetShape();
			// return new Shape(argumentShape->kind, argumentShape->size);
		} 
		case BuiltinMethod::Kind::Less:
		case BuiltinMethod::Kind::Greater:
		case BuiltinMethod::Kind::LessEqual:
		case BuiltinMethod::Kind::GreaterEqual:
		case BuiltinMethod::Kind::Equal:
		case BuiltinMethod::Kind::NotEqual:
		case BuiltinMethod::Kind::Plus:
		case BuiltinMethod::Kind::Minus:
		case BuiltinMethod::Kind::Multiply:
		case BuiltinMethod::Kind::Divide:
		case BuiltinMethod::Kind::Power:
		case BuiltinMethod::Kind::LogarithmBase:
		case BuiltinMethod::Kind::Modulo:
		case BuiltinMethod::Kind::And:
		case BuiltinMethod::Kind::Or:
		case BuiltinMethod::Kind::Nand:
		case BuiltinMethod::Kind::Nor:
		case BuiltinMethod::Kind::Xor:
		{
			// const auto argumentShape1 = arguments.at(0)->GetShape();
			// const auto argumentShape2 = arguments.at(1)->GetShape();

			// Shape::Kind kind = Shape::Kind::Vector;
			// long size = 0;

			// if (argumentShape1->size == argumentShape2->size)
			// {
			// 	size = argumentShape1->size;
			// }
			// else if (argumentShape1->size == 1)
			// {
			// 	size = argumentShape2->size;
			// }
			// else if (argumentShape2->size == 1)
			// {
			// 	size = argumentShape1->size;
			// }
			// else
			// {
			// 	Utils::Logger::LogError("Dyadic elementwise functions cannot be vectors of different sizes [" + std::to_string(argumentShape1->size) + " != " + std::to_string(argumentShape2->size) + "]");
			// }

			// return new Shape(kind, size);
		}
		case BuiltinMethod::Kind::Compress:
			//TODO: Should set some type of a reference variable to the input shape
			return new Shape(Shape::Kind::Vector, Shape::DynamicSize);
		case BuiltinMethod::Kind::Count:
		case BuiltinMethod::Kind::Sum:
		case BuiltinMethod::Kind::Average:
		case BuiltinMethod::Kind::Minimum:
		case BuiltinMethod::Kind::Maximum:
			return new Shape(Shape::Kind::Vector, 1);
			//TODO: @vector
		// case BuiltinMethod::Kind::Fill:
		// 	return new Shape(Shape::Kind::Vector, static_cast<Literal<int64_t> *>(arguments.at(0))->GetValue(0));
		default:
			Utils::Logger::LogError("Shape analysis does not support builtin method " + method->GetName());
	}
}

void ShapeAnalysis::Visit(CastExpression *cast)
{
	// cast->SetShape(cast->GetExpression()->GetShape());
}

void ShapeAnalysis::Visit(Identifier *identifier)
{
	// identifier->SetShape(identifier->GetDeclaration()->GetShape());
}

// void ShapeAnalysis::Visit(Literal<int64_t> *literal)
// {
// 	// literal->SetShape(new Shape(Shape::Kind::Vector, literal->GetCount()));
// }

// void ShapeAnalysis::Visit(Literal<double> *literal)
// {
// 	// literal->SetShape(new Shape(Shape::Kind::Vector, literal->GetCount()));
// }

// void ShapeAnalysis::Visit(Literal<std::string> *literal)
// {
// 	// literal->SetShape(new Shape(Shape::Kind::Vector, literal->GetCount()));
// }

// void ShapeAnalysis::Visit(Symbol *symbol)
// {
// 	// symbol->SetShape(new Shape(Shape::Kind::Vector, symbol->GetCount()));
// }

}
