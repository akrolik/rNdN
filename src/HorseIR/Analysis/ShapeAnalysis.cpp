#include "HorseIR/Analysis/Shape/ShapeAnalysis.h"

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

void ShapeAnalysis::Analyze(const MethodDeclaration *method)
{
	m_shapes.push(std::make_tuple(nullptr, new MethodInvocationShapes()));
	method->Accept(*this);

	auto& context = m_shapes.top();
	m_results->AddInvocationShapes(std::get<0>(context), std::get<1>(context));
	m_shapes.pop();
}

Shape *ShapeAnalysis::GetShape(const Expression *expression)
{
	return std::get<1>(m_shapes.top())->GetShape(expression);
}

Shape *ShapeAnalysis::GetShape(const std::string& variable)
{
	return std::get<1>(m_shapes.top())->GetShape(variable);
}

void ShapeAnalysis::SetShape(const Expression *expression, Shape *shape)
{
	std::get<1>(m_shapes.top())->SetShape(expression, shape);
}

void ShapeAnalysis::SetShape(const std::string& variable, Shape *shape)
{
	std::get<1>(m_shapes.top())->SetShape(variable, shape);
}

void ShapeAnalysis::Visit(const Parameter *parameter)
{
	// Check to make sure the correct input shapes have been set

	GetShape(parameter->GetName());
}

void ShapeAnalysis::Visit(const AssignStatement *assign)
{
	// Traverse all children of the assignment

	ConstForwardTraversal::Visit(assign);

	// Update the declaration with the shape propagated from the expression

	auto declaration = assign->GetDeclaration();
	auto expression = assign->GetExpression();

        SetShape(declaration->GetName(), GetShape(expression));
}

void ShapeAnalysis::Visit(const ReturnStatement *ret)
{
	std::get<1>(m_shapes.top())->SetReturnShape(GetShape(ret->GetIdentifier()->GetString()));
}

void ShapeAnalysis::Visit(const CallExpression *call)
{
	// Collect shape information for the arguments

	ConstForwardTraversal::Visit(call);

	// Store the current call to give context to the dynamic sizes

	m_call = call;

	// Analyze the function according to the shape rules

	SetShape(call, AnalyzeCall(call->GetMethod(), call->GetArguments()));

	// Reset current expression to the enclosing call

	m_call = std::get<0>(m_shapes.top());
}

[[noreturn]] void ShapeAnalysis::ShapeError(const MethodDeclaration *method, const std::vector<Expression *>& arguments)
{
	std::string message = "Incompatible shapes [";
	bool first = true;
	for (const auto& argument : arguments)
	{
		if (!first)
		{
			message += ", ";
		}
		first = false;
		message += GetShape(argument)->ToString();
	}
	message += "] to function '" + method->GetName() + "'";
	Utils::Logger::LogError(message);
}

Shape *ShapeAnalysis::AnalyzeCall(const MethodDeclaration *method, const std::vector<Expression *>& arguments)
{
	switch (method->GetKind())
	{
		case MethodDeclaration::Kind::Builtin:
			return AnalyzeCall(static_cast<const BuiltinMethod *>(method), arguments);
		case MethodDeclaration::Kind::Definition:
			return AnalyzeCall(static_cast<const Method *>(method), arguments);
		default:
			Utils::Logger::LogError("Unsupported method kind");
	}
}

Shape *ShapeAnalysis::AnalyzeCall(const Method *method, const std::vector<Expression *>& arguments)
{
	// Create a new shape mapping for this invocation and set all parameters

	auto localShapes = new MethodInvocationShapes();

	unsigned int i = 0;
	auto& parameters = method->GetParameters();
	for (auto& argument : arguments)
	{
		auto name = parameters.at(i);
		localShapes->SetShape(name->GetName(), GetShape(argument));
		++i;
	}

	// Update the scope for the new context

	m_shapes.push(std::make_tuple(m_call, localShapes));

	ConstForwardTraversal::Visit(method);

	// Finalize the invocation shape map
	m_results->AddInvocationShapes(std::get<0>(m_shapes.top()), std::get<1>(m_shapes.top()));
	m_shapes.pop();

	return localShapes->GetReturnShape();
}

Shape *ShapeAnalysis::AnalyzeCall(const BuiltinMethod *method, const std::vector<Expression *>& arguments)
{
	switch (method->GetKind())
	{
#define Require(x) if (!(x)) break

		case BuiltinMethod::Kind::Absolute:
		case BuiltinMethod::Kind::Negate:
		case BuiltinMethod::Kind::Ceiling:
		case BuiltinMethod::Kind::Floor:
		case BuiltinMethod::Kind::Round:
		case BuiltinMethod::Kind::Conjugate:
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
		case BuiltinMethod::Kind::Date:
		case BuiltinMethod::Kind::DateYear:
		case BuiltinMethod::Kind::DateMonth:
		case BuiltinMethod::Kind::DateDay:
		case BuiltinMethod::Kind::Time:
		case BuiltinMethod::Kind::TimeHour:
		case BuiltinMethod::Kind::TimeMinute:
		case BuiltinMethod::Kind::TimeSecond:
		case BuiltinMethod::Kind::TimeMillisecond:
		{
			auto argumentShape = GetShape(arguments.at(0));
			Require(IsShape<VectorShape>(argumentShape));
			return argumentShape;
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
		case BuiltinMethod::Kind::DatetimeDifference:
		{
			auto argumentShape1 = GetShape(arguments.at(0));
			auto argumentShape2 = GetShape(arguments.at(1));
			Require(IsShape<VectorShape>(argumentShape1) && IsShape<VectorShape>(argumentShape2));

			auto argumentSize1 = HorseIR::GetShape<VectorShape>(argumentShape1)->GetSize();
			auto argumentSize2 = HorseIR::GetShape<VectorShape>(argumentShape2)->GetSize();

			const Shape::Size *size = nullptr;

			if (argumentSize1->m_kind == Shape::Size::Kind::Constant && static_cast<const Shape::ConstantSize *>(argumentSize1)->m_value == 1)
			{
				size = argumentSize2;
			}
			else if (argumentSize2->m_kind == Shape::Size::Kind::Constant && static_cast<const Shape::ConstantSize *>(argumentSize2)->m_value == 1)
			{
				size = argumentSize1;
			}
			else if (argumentSize1->m_kind == Shape::Size::Kind::Dynamic || argumentSize2->m_kind == Shape::Size::Kind::Dynamic)
			{
				size = new Shape::DynamicSize(std::get<0>(m_shapes.top()), m_call);
			}
			else if (argumentSize1->m_kind == Shape::Size::Kind::Symbol || argumentSize2->m_kind == Shape::Size::Kind::Symbol)
			{
				if (*argumentSize1 != *argumentSize2)
				{
					size = new Shape::DynamicSize(std::get<0>(m_shapes.top()), m_call);
				}
				else
				{
					size = argumentSize1;
				}
			}
			else
			{
				auto constant1 = static_cast<const Shape::ConstantSize *>(argumentSize1)->m_value;
				auto constant2 = static_cast<const Shape::ConstantSize *>(argumentSize2)->m_value;

				if (constant1 != constant2)
				{
					Utils::Logger::LogError("Dyadic elementwise functions cannot be vectors of different sizes [" + std::to_string(constant1) + " != " + std::to_string(constant2) + "]");
				}
				size = new Shape::ConstantSize(constant1);
			}
			return new VectorShape(size);
		}
		case BuiltinMethod::Kind::Compress:
		{
			auto argumentShape1 = GetShape(arguments.at(0));
			auto argumentShape2 = GetShape(arguments.at(1));
			Require(IsShape<VectorShape>(argumentShape1) && IsShape<VectorShape>(argumentShape2));

			auto argumentSize1 = HorseIR::GetShape<VectorShape>(argumentShape1)->GetSize();
			auto argumentSize2 = HorseIR::GetShape<VectorShape>(argumentShape2)->GetSize();
			//TODO: Better handling of dynamics
			Require(*argumentSize1 == *argumentSize2 || argumentSize1->m_kind == Shape::Size::Kind::Dynamic || argumentSize2->m_kind == Shape::Size::Kind::Dynamic);

			return new VectorShape(new Shape::CompressedSize(std::get<0>(m_shapes.top()), arguments.at(0), argumentSize2));
		}
		// @count and @len are aliases
		case BuiltinMethod::Kind::Length:
		case BuiltinMethod::Kind::Count:
		case BuiltinMethod::Kind::Sum:
		case BuiltinMethod::Kind::Average:
		case BuiltinMethod::Kind::Minimum:
		case BuiltinMethod::Kind::Maximum:
		{
			auto argumentShape = GetShape(arguments.at(0));
			Require(IsShape<VectorShape>(argumentShape));
			return new VectorShape(new Shape::ConstantSize(1));
		}
		case BuiltinMethod::Kind::List:
		{
			const Shape *shape = nullptr;
			for (const auto& argument : arguments)
			{
				auto argumentShape = GetShape(argument);
				if (shape == nullptr)
				{
					shape = argumentShape;
				}
				else if (*shape != *argumentShape)
				{
					//TODO: Implement full @list
					break;
				}
			}
			return new ListShape(new Shape::ConstantSize(arguments.size()), shape);
		}
		case BuiltinMethod::Kind::Enlist:
		{
			auto argumentShape = GetShape(arguments.at(0));
			return new ListShape(new Shape::ConstantSize(1), argumentShape);
		}
		case BuiltinMethod::Kind::Table:
		{
			auto argumentShape1 = GetShape(arguments.at(0));
			auto argumentShape2 = GetShape(arguments.at(1));
			Require(IsShape<VectorShape>(argumentShape1) && IsShape<ListShape>(argumentShape2));

			auto listShape = HorseIR::GetShape<ListShape>(argumentShape2);
			auto elementShape = listShape->GetElementShape();
			Require(IsShape<VectorShape>(elementShape));

			auto argumentSize1 = HorseIR::GetShape<VectorShape>(argumentShape1)->GetSize();
			auto argumentSize2 = listShape->GetListSize();
			Require(*argumentSize1 == *argumentSize2);

			return new TableShape(argumentSize1, HorseIR::GetShape<VectorShape>(elementShape)->GetSize());
		}
		case BuiltinMethod::Kind::ColumnValue:
		{
			auto argumentShape = GetShape(arguments.at(0));
			Require(IsShape<TableShape>(argumentShape));

			auto tableShape = HorseIR::GetShape<TableShape>(argumentShape);
			return new VectorShape(tableShape->GetRowsSize());
		}
		case BuiltinMethod::Kind::LoadTable:
		{
			return new TableShape(new Shape::ConstantSize(0), new Shape::SymbolSize(static_cast<const SymbolLiteral *>(arguments.at(0))->GetValue(0)));
		}
		default:
		{
			Utils::Logger::LogError("Shape analysis is not supported for builtin method '" + method->GetName() + "'");
		}
	}

	ShapeError(method, arguments);
}

void ShapeAnalysis::Visit(const CastExpression *cast)
{
	// Traverse the expression

	ConstForwardTraversal::Visit(cast);

	// Propagate the shape from the expression to the cast

	SetShape(cast, GetShape(cast->GetExpression()));
}

void ShapeAnalysis::Visit(const Identifier *identifier)
{
	SetShape(identifier, GetShape(identifier->GetDeclaration()->GetName()));
}

void ShapeAnalysis::Visit(const BoolLiteral *literal)
{
	SetShape(literal, new VectorShape(new Shape::ConstantSize(literal->GetCount())));
}

void ShapeAnalysis::Visit(const Int8Literal *literal)
{
	SetShape(literal, new VectorShape(new Shape::ConstantSize(literal->GetCount())));
}

void ShapeAnalysis::Visit(const Int16Literal *literal)
{
	SetShape(literal, new VectorShape(new Shape::ConstantSize(literal->GetCount())));
}

void ShapeAnalysis::Visit(const Int32Literal *literal)
{
	SetShape(literal, new VectorShape(new Shape::ConstantSize(literal->GetCount())));
}

void ShapeAnalysis::Visit(const Int64Literal *literal)
{
	SetShape(literal, new VectorShape(new Shape::ConstantSize(literal->GetCount())));
}

void ShapeAnalysis::Visit(const Float32Literal *literal)
{
	SetShape(literal, new VectorShape(new Shape::ConstantSize(literal->GetCount())));
}

void ShapeAnalysis::Visit(const Float64Literal *literal)
{
	SetShape(literal, new VectorShape(new Shape::ConstantSize(literal->GetCount())));
}

void ShapeAnalysis::Visit(const StringLiteral *literal)
{
	SetShape(literal, new VectorShape(new Shape::ConstantSize(literal->GetCount())));
}

void ShapeAnalysis::Visit(const SymbolLiteral *literal)
{
	SetShape(literal, new VectorShape(new Shape::ConstantSize(literal->GetCount())));
}

void ShapeAnalysis::Visit(const DateLiteral *literal)
{
	SetShape(literal, new VectorShape(new Shape::ConstantSize(literal->GetCount())));
}

void ShapeAnalysis::Visit(const FunctionLiteral *literal)
{
	// Function literals have only 1 element

	SetShape(literal, new VectorShape(new Shape::ConstantSize(1)));
}

}
