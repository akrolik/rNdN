#include "HorseIR/ShapeAnalysis.h"

#include "HorseIR/BuiltinFunctions.h"
#include "HorseIR/Tree/Method.h"
#include "HorseIR/Tree/Parameter.h"
#include "HorseIR/Tree/Expressions/CallExpression.h"
#include "HorseIR/Tree/Expressions/CastExpression.h"
#include "HorseIR/Tree/Expressions/Expression.h"
#include "HorseIR/Tree/Expressions/Identifier.h"
#include "HorseIR/Tree/Expressions/Literal.h"
#include "HorseIR/Tree/Statements/AssignStatement.h"

#include <iostream>

namespace HorseIR {

void ShapeAnalysis::SetInputShape(const Parameter *parameter, Shape *shape)
{
	SetShape(parameter->GetName(), shape);
}

void ShapeAnalysis::Analyze(HorseIR::Method *method)
{
	method->Accept(*this);
}

const Shape *ShapeAnalysis::GetShape(const std::string& identifier) const
{
	return m_identifierMap.at(identifier);
}

void ShapeAnalysis::Visit(Parameter *parameter)
{
	// Check to make sure the correct input shapes have been set

	if (m_identifierMap.find(parameter->GetName()) == m_identifierMap.end())
	{
		std::cerr << "[ERROR] Shape analysis missing input shape for parameter " << parameter->GetName() << std::endl;
		std::exit(EXIT_FAILURE);
	}
}

void ShapeAnalysis::Visit(AssignStatement *assign)
{
	// Traverse all children of the assignment

	ForwardTraversal::Visit(assign);

	// Update the map with the RHS expression shape

	SetShape(assign->GetTargetName(), GetShape(assign->GetExpression()));
}

void ShapeAnalysis::Visit(CallExpression *call)
{
	// Collect shape information for the arguments

	ForwardTraversal::Visit(call);

	// Analyze the function according to the shape rules

	BuiltinFunction function = GetBuiltinFunction(call->GetName());
	switch (function)
	{
		case BuiltinFunction::Absolute:
		case BuiltinFunction::Negate:
		case BuiltinFunction::Ceiling:
		case BuiltinFunction::Floor:
		case BuiltinFunction::Round:
		case BuiltinFunction::Reciprocal:
		case BuiltinFunction::Sign:
		case BuiltinFunction::Pi:
		case BuiltinFunction::Not:
		case BuiltinFunction::Logarithm:
		case BuiltinFunction::Exponential:
		case BuiltinFunction::Cosine:
		case BuiltinFunction::Sine:
		case BuiltinFunction::Tangent:
		case BuiltinFunction::InverseCosine:
		case BuiltinFunction::InverseSine:
		case BuiltinFunction::InverseTangent:
		case BuiltinFunction::HyperbolicCosine:
		case BuiltinFunction::HyperbolicSine:
		case BuiltinFunction::HyperbolicTangent:
		case BuiltinFunction::HyperbolicInverseCosine:
		case BuiltinFunction::HyperbolicInverseSine:
		case BuiltinFunction::HyperbolicInverseTangent:
		{
			const Shape *arg = GetShape(call->GetArgument(0));
			SetShape(call,  new Shape(arg->kind, arg->size));
			break;
		}
		case BuiltinFunction::Less:
		case BuiltinFunction::Greater:
		case BuiltinFunction::LessEqual:
		case BuiltinFunction::GreaterEqual:
		case BuiltinFunction::Equal:
		case BuiltinFunction::NotEqual:
		case BuiltinFunction::Plus:
		case BuiltinFunction::Minus:
		case BuiltinFunction::Multiply:
		case BuiltinFunction::Divide:
		case BuiltinFunction::Power:
		case BuiltinFunction::Logarithm2:
		case BuiltinFunction::Modulo:
		case BuiltinFunction::And:
		case BuiltinFunction::Or:
		case BuiltinFunction::Nand:
		case BuiltinFunction::Nor:
		case BuiltinFunction::Xor:
		{
			const Shape *arg1 = GetShape(call->GetArgument(0));
			const Shape *arg2 = GetShape(call->GetArgument(1));

			Shape::Kind kind = Shape::Kind::Vector;
			long size = 0;

			if (arg1->size == arg2->size)
			{
				size = arg1->size;
			}
			else if (arg1->size == 1)
			{
				size = arg2->size;
			}
			else if (arg2->size == 1)
			{
				size = arg1->size;
			}
			else
			{
				std::cerr << "[ERROR] Dyadic elementwise functions cannot be vectors of different sizes [" << arg1->size << " != " << arg2->size << "]" << std::endl;
				std::exit(EXIT_FAILURE);
			}

			SetShape(call, new Shape(kind, size));
			break;
		}
		case BuiltinFunction::Compress:
		{
			//TODO: Should set some type of a reference variable to the input shape
			SetShape(call, new Shape(Shape::Kind::Vector, Shape::DynamicSize));
			break;
		}
		case BuiltinFunction::Count:
		case BuiltinFunction::Sum:
		case BuiltinFunction::Average:
		case BuiltinFunction::Minimum:
		case BuiltinFunction::Maximum:
		{
			SetShape(call, new Shape(Shape::Kind::Vector, 1));
			break;
		}
		case BuiltinFunction::Fill:
		{
			//TODO: Fill shape analysis
			// SetShape(call, new Shape(Shape::Kind::Vector, call->GetArgument(0)));
			break;
		}
		case BuiltinFunction::Conjugate:
		case BuiltinFunction::Unsupported:
			std::cerr << "[ERROR] Shape analysis does not support function " << call->GetName() << std::endl;
			std::exit(EXIT_FAILURE);
	}
}

void ShapeAnalysis::Visit(CastExpression *cast)
{
	SetShape(cast, GetShape(cast->GetExpression()));
}

void ShapeAnalysis::Visit(Identifier *identifier)
{
	SetShape(identifier, GetShape(identifier->GetString()));
}

void ShapeAnalysis::Visit(Literal<int64_t> *literal)
{
	SetShape(literal, new Shape(Shape::Kind::Vector, literal->GetCount()));
}

void ShapeAnalysis::Visit(Literal<double> *literal)
{
	SetShape(literal, new Shape(Shape::Kind::Vector, literal->GetCount()));
}

void ShapeAnalysis::Visit(Literal<std::string> *literal)
{
	SetShape(literal, new Shape(Shape::Kind::Vector, literal->GetCount()));
}

void ShapeAnalysis::Visit(Symbol *symbol)
{
	//TODO: ShapeAnalysis for symbols
}

const Shape *ShapeAnalysis::GetShape(const Expression *expression) const
{
	if (m_expressionMap.find(expression) == m_expressionMap.end())
	{
		std::cerr << "[ERROR] Shape not found for expression " << expression->ToString() << std::endl;
		std::exit(EXIT_FAILURE);
	}
	return m_expressionMap.at(expression);
}

void ShapeAnalysis::SetShape(const Expression *expression, const Shape *shape)
{
	m_expressionMap.insert({expression, shape});
}

void ShapeAnalysis::SetShape(const std::string& identifier, const Shape *shape)
{
	m_identifierMap.insert({identifier, shape});
}

void ShapeAnalysis::Dump() const
{
	std::cout << "[DEBUG] Shape Analysis Dump" << std::endl;
	std::cout << "------------------------------" << std::endl;
	for (auto it = m_identifierMap.cbegin(); it != m_identifierMap.cend(); ++it)
	{
		std::cout << "  " << it->first << ": " << it->second->ToString() << std::endl;
	}
}

}
