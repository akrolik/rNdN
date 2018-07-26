#include "HorseIR/ShapeAnalysis.h"

#include "HorseIR/BuiltinFunctions.h"
#include "HorseIR/Tree/Method.h"
#include "HorseIR/Tree/Parameter.h"
#include "HorseIR/Tree/Expressions/CallExpression.h"
#include "HorseIR/Tree/Expressions/Expression.h"
#include "HorseIR/Tree/Expressions/Identifier.h"
#include "HorseIR/Tree/Statements/AssignStatement.h"

#include <iostream>

namespace HorseIR {

void ShapeAnalysis::SetInputShape(const Parameter *parameter, Shape *shape)
{
	m_identifierMap.insert({parameter->GetName(), shape});
}

void ShapeAnalysis::Analyze(HorseIR::Method *method)
{
	m_identifierMap.clear();
	method->Accept(*this);
}

Shape *ShapeAnalysis::GetShape(const std::string& identifier) const
{
	return m_identifierMap.at(identifier);
}

void ShapeAnalysis::Visit(Parameter *parameter)
{
	if (m_identifierMap.find(parameter->GetName()) == m_identifierMap.end())
	{
		std::cerr << "[ERROR] Shape analysis missing input shape for parameter " << parameter->GetName() << std::endl;
		std::exit(EXIT_FAILURE);
	}
}

void ShapeAnalysis::Visit(AssignStatement *assign)
{
	m_identifierMap.insert({assign->GetTargetName(), GetExpressionShape(assign->GetExpression())});
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
			Shape *arg = m_expressionMap.at(call->GetArgument(0));
			m_expressionMap.insert({call, new Shape(arg->kind, arg->size)});
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
			Shape *arg1 = m_expressionMap.at(call->GetArgument(0));
			Shape *arg2 = m_expressionMap.at(call->GetArgument(1));

			Shape::Kind kind = Shape::Kind::Vector;
			unsigned int size = 0;

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

			m_expressionMap.insert({call, new Shape(kind, size)});
			break;
		}
		case BuiltinFunction::Compress:
		{
			//TODO: Compression shape analysis
			break;
		}
		case BuiltinFunction::Count:
		case BuiltinFunction::Sum:
		case BuiltinFunction::Average:
		case BuiltinFunction::Minimum:
		case BuiltinFunction::Maximum:
		{
			m_expressionMap.insert({call, new Shape(Shape::Kind::Vector, 1)});
			break;
		}
		case BuiltinFunction::Fill:
		{
			//TODO: Fill shape analysis
			// m_shape = new Shape(Shape::Kind::Vector, 1);
			break;
		}
		case BuiltinFunction::Conjugate:
		case BuiltinFunction::Unsupported:
			std::cerr << "[ERROR] Shape analysis does not support function " << call->GetName() << std::endl;
			std::exit(EXIT_FAILURE);
	}
}

Shape *ShapeAnalysis::GetExpressionShape(Expression *expression)
{
	if (m_expressionMap.find(expression) == m_expressionMap.end())
	{
		std::cerr << "[ERROR] Shape not found for expression " << expression->ToString() << std::endl;
		std::exit(EXIT_FAILURE);
	}
	return m_expressionMap.at(expression);
}

}
