#include "Analysis/Shape/ShapeAnalysisHelper.h"

#include "Analysis/Shape/ShapeUtils.h"

namespace Analysis {

std::vector<const Shape *> ShapeAnalysisHelper::GetShapes(const ShapeAnalysis::Properties& properties, const HorseIR::Expression *expression)
{
	ShapeAnalysisHelper helper(properties);
	expression->Accept(helper);
	return helper.GetShapes(expression);
}

void ShapeAnalysisHelper::Visit(const HorseIR::CallExpression *call)
{
	// Collect shape information for the arguments

	for (const auto& argument : call->GetArguments())
	{
		argument->Accept(*this);
	}

	// Analyze the function according to the shape rules. Store the call for dynamic sizes

	m_call = call;
	SetShapes(call, AnalyzeCall(call->GetFunctionLiteral()->GetFunction(), call->GetArguments()));
	m_call = nullptr;
}

std::vector<const Shape *> ShapeAnalysisHelper::AnalyzeCall(const HorseIR::FunctionDeclaration *function, const std::vector<HorseIR::Operand *>& arguments)
{
	switch (function->GetKind())
	{
		case HorseIR::FunctionDeclaration::Kind::Builtin:
			return AnalyzeCall(static_cast<const HorseIR::BuiltinFunction *>(function), arguments);
		case HorseIR::FunctionDeclaration::Kind::Definition:
			return AnalyzeCall(static_cast<const HorseIR::Function *>(function), arguments);
		default:
			Utils::Logger::LogError("Unsupported function kind");
	}
}

std::vector<const Shape *> ShapeAnalysisHelper::AnalyzeCall(const HorseIR::Function *function, const std::vector<HorseIR::Operand *>& arguments)
{
	//TODO: Make a more specific wildcard type based on the type

	return {new WildcardShape()};
}

std::vector<const Shape *> ShapeAnalysisHelper::AnalyzeCall(const HorseIR::BuiltinFunction *function, const std::vector<HorseIR::Operand *>& arguments)
{
	switch (function->GetPrimitive())
	{
#define Require(x) if (!(x)) break

		case HorseIR::BuiltinFunction::Primitive::Absolute:
		case HorseIR::BuiltinFunction::Primitive::Negate:
		case HorseIR::BuiltinFunction::Primitive::Ceiling:
		case HorseIR::BuiltinFunction::Primitive::Floor:
		case HorseIR::BuiltinFunction::Primitive::Round:
		case HorseIR::BuiltinFunction::Primitive::Conjugate:
		case HorseIR::BuiltinFunction::Primitive::Reciprocal:
		case HorseIR::BuiltinFunction::Primitive::Sign:
		case HorseIR::BuiltinFunction::Primitive::Pi:
		case HorseIR::BuiltinFunction::Primitive::Not:
		case HorseIR::BuiltinFunction::Primitive::Logarithm:
		case HorseIR::BuiltinFunction::Primitive::Logarithm2:
		case HorseIR::BuiltinFunction::Primitive::Logarithm10:
		case HorseIR::BuiltinFunction::Primitive::SquareRoot:
		case HorseIR::BuiltinFunction::Primitive::Exponential:
		case HorseIR::BuiltinFunction::Primitive::Cosine:
		case HorseIR::BuiltinFunction::Primitive::Sine:
		case HorseIR::BuiltinFunction::Primitive::Tangent:
		case HorseIR::BuiltinFunction::Primitive::InverseCosine:
		case HorseIR::BuiltinFunction::Primitive::InverseSine:
		case HorseIR::BuiltinFunction::Primitive::InverseTangent:
		case HorseIR::BuiltinFunction::Primitive::HyperbolicCosine:
		case HorseIR::BuiltinFunction::Primitive::HyperbolicSine:
		case HorseIR::BuiltinFunction::Primitive::HyperbolicTangent:
		case HorseIR::BuiltinFunction::Primitive::HyperbolicInverseCosine:
		case HorseIR::BuiltinFunction::Primitive::HyperbolicInverseSine:
		case HorseIR::BuiltinFunction::Primitive::HyperbolicInverseTangent:
		case HorseIR::BuiltinFunction::Primitive::Date:
		case HorseIR::BuiltinFunction::Primitive::DateYear:
		case HorseIR::BuiltinFunction::Primitive::DateMonth:
		case HorseIR::BuiltinFunction::Primitive::DateDay:
		case HorseIR::BuiltinFunction::Primitive::Time:
		case HorseIR::BuiltinFunction::Primitive::TimeHour:
		case HorseIR::BuiltinFunction::Primitive::TimeMinute:
		case HorseIR::BuiltinFunction::Primitive::TimeSecond:
		case HorseIR::BuiltinFunction::Primitive::TimeMillisecond:
		{
			auto argumentShape = GetShape(arguments.at(0));
			Require(IsShape<VectorShape>(argumentShape));
			return {argumentShape};
		}
		case HorseIR::BuiltinFunction::Primitive::Less:
		case HorseIR::BuiltinFunction::Primitive::Greater:
		case HorseIR::BuiltinFunction::Primitive::LessEqual:
		case HorseIR::BuiltinFunction::Primitive::GreaterEqual:
		case HorseIR::BuiltinFunction::Primitive::Equal:
		case HorseIR::BuiltinFunction::Primitive::NotEqual:
		case HorseIR::BuiltinFunction::Primitive::Plus:
		case HorseIR::BuiltinFunction::Primitive::Minus:
		case HorseIR::BuiltinFunction::Primitive::Multiply:
		case HorseIR::BuiltinFunction::Primitive::Divide:
		case HorseIR::BuiltinFunction::Primitive::Power:
		case HorseIR::BuiltinFunction::Primitive::LogarithmBase:
		case HorseIR::BuiltinFunction::Primitive::Modulo:
		case HorseIR::BuiltinFunction::Primitive::And:
		case HorseIR::BuiltinFunction::Primitive::Or:
		case HorseIR::BuiltinFunction::Primitive::Nand:
		case HorseIR::BuiltinFunction::Primitive::Nor:
		case HorseIR::BuiltinFunction::Primitive::Xor:
		case HorseIR::BuiltinFunction::Primitive::DatetimeDifference:
		{
			auto argumentShape1 = GetShape(arguments.at(0));
			auto argumentShape2 = GetShape(arguments.at(1));
			Require(IsShape<VectorShape>(argumentShape1) && IsShape<VectorShape>(argumentShape2));

			auto argumentSize1 = CastShape<VectorShape>(argumentShape1)->GetSize();
			auto argumentSize2 = CastShape<VectorShape>(argumentShape2)->GetSize();

			const Shape::Size *size = nullptr;

			if (argumentSize1->m_kind == Shape::Size::Kind::Constant && static_cast<const Shape::ConstantSize *>(argumentSize1)->m_value == 1)
			{
				size = argumentSize2;
			}
			else if (argumentSize2->m_kind == Shape::Size::Kind::Constant && static_cast<const Shape::ConstantSize *>(argumentSize2)->m_value == 1)
			{
				size = argumentSize1;
			}
			else if (argumentSize1->m_kind == Shape::Size::Kind::Constant && argumentSize2->m_kind == Shape::Size::Kind::Constant)
			{
				auto constant1 = static_cast<const Shape::ConstantSize *>(argumentSize1)->m_value;
				auto constant2 = static_cast<const Shape::ConstantSize *>(argumentSize2)->m_value;

				if (constant1 != constant2)
				{
					Utils::Logger::LogError("Dyadic elementwise function '" + function->GetName() + "' requires vector of same length (or broadcast) [" + std::to_string(constant1) + " != " + std::to_string(constant2) + "]");
				}
				size = new Shape::ConstantSize(constant1);
			}
			else
			{
				if (*argumentSize1 != *argumentSize2)
				{
					size = new Shape::DynamicSize(m_call);
				}
				else
				{
					size = argumentSize1;
				}
			}
			return {new VectorShape(size)};
		}
		case HorseIR::BuiltinFunction::Primitive::Unique:
		{
			auto argumentShape = GetShape(arguments.at(0));
			Require(IsShape<VectorShape>(argumentShape));
			return {new VectorShape(new Shape::DynamicSize(m_call))};
		}
		case HorseIR::BuiltinFunction::Primitive::String:
		{
			return {new VectorShape(new Shape::ConstantSize(1))};
		}
		case HorseIR::BuiltinFunction::Primitive::Range:
		{
			//TODO:
			break;
		}
		case HorseIR::BuiltinFunction::Primitive::Factorial:
		case HorseIR::BuiltinFunction::Primitive::Reverse:
		{
			auto argumentShape = GetShape(arguments.at(0));
			Require(IsShape<VectorShape>(argumentShape));
			return {argumentShape};
		}
		case HorseIR::BuiltinFunction::Primitive::Where:
		{
			//TODO:
			break;
		}
		case HorseIR::BuiltinFunction::Primitive::Group:
		{
			//TODO:
			break;
		}
		case HorseIR::BuiltinFunction::Primitive::Append:
		{
			//TODO:
			break;
		}
		case HorseIR::BuiltinFunction::Primitive::Like:
		{
			//TODO:
			break;
		}
		case HorseIR::BuiltinFunction::Primitive::Compress:
		{
			auto argumentShape1 = GetShape(arguments.at(0));
			auto argumentShape2 = GetShape(arguments.at(1));
			Require(IsShape<VectorShape>(argumentShape1) && IsShape<VectorShape>(argumentShape2));

			auto argumentSize1 = CastShape<VectorShape>(argumentShape1)->GetSize();
			auto argumentSize2 = CastShape<VectorShape>(argumentShape2)->GetSize();

			auto dynamic1 = (argumentSize1->m_kind == Shape::Size::Kind::Dynamic);
			auto dynamic2 = (argumentSize2->m_kind == Shape::Size::Kind::Dynamic);
			Require(*argumentSize1 == *argumentSize2 || dynamic1 || dynamic2);

			const Shape::Size *size = nullptr;
			if (dynamic1 && dynamic2)
			{
				size = new Shape::DynamicSize(m_call);
			}
			else if (dynamic1)
			{
				size = argumentSize2;
			}
			else
			{
				size = argumentSize1;
			}
			return {new VectorShape(new Shape::CompressedSize(arguments.at(0), size))};
		}
		case HorseIR::BuiltinFunction::Primitive::Random_k:
		{
			//TODO:
			break;
		}
		case HorseIR::BuiltinFunction::Primitive::IndexOf:
		{
			//TODO:
			break;
		}
		case HorseIR::BuiltinFunction::Primitive::Take:
		{
			//TODO:
			break;
		}
		case HorseIR::BuiltinFunction::Primitive::Drop:
		{
			//TODO:
			break;
		}
		case HorseIR::BuiltinFunction::Primitive::Order:
		{
			//TODO:
			break;
		}
		case HorseIR::BuiltinFunction::Primitive::Member:
		{
			//TODO:
			break;
		}
		case HorseIR::BuiltinFunction::Primitive::Vector:
		{
			//TODO:
			break;
		}
		// @count and @len are aliases
		case HorseIR::BuiltinFunction::Primitive::Length:
		case HorseIR::BuiltinFunction::Primitive::Count:
		case HorseIR::BuiltinFunction::Primitive::Sum:
		case HorseIR::BuiltinFunction::Primitive::Average:
		case HorseIR::BuiltinFunction::Primitive::Minimum:
		case HorseIR::BuiltinFunction::Primitive::Maximum:
		{
			auto argumentShape = GetShape(arguments.at(0));
			Require(IsShape<VectorShape>(argumentShape));
			return {new VectorShape(new Shape::ConstantSize(1))};
		}
		case HorseIR::BuiltinFunction::Primitive::Raze:
		{
			//TODO:
			break;
		}
		case HorseIR::BuiltinFunction::Primitive::List:
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
					shape = new WildcardShape();
					break;
				}
			}
			return {new ListShape(new Shape::ConstantSize(arguments.size()), shape)};
		}
		case HorseIR::BuiltinFunction::Primitive::ToList:
		{
			//TODO:
			break;
		}
		case HorseIR::BuiltinFunction::Primitive::Each:
		{
			//TODO:
			break;
		}
		case HorseIR::BuiltinFunction::Primitive::EachItem:
		{
			//TODO:
			break;
		}
		case HorseIR::BuiltinFunction::Primitive::EachLeft:
		{
			//TODO:
			break;
		}
		case HorseIR::BuiltinFunction::Primitive::EachRight:
		{
			//TODO:
			break;
		}
		case HorseIR::BuiltinFunction::Primitive::Match:
		{
			//TODO:
			break;
		}
		case HorseIR::BuiltinFunction::Primitive::Enum:
		{
			//TODO:
			break;
		}
		case HorseIR::BuiltinFunction::Primitive::Dictionary:
		{
			//TODO:
			break;
		}
		case HorseIR::BuiltinFunction::Primitive::Table:
		{
			auto argumentShape1 = GetShape(arguments.at(0));
			auto argumentShape2 = GetShape(arguments.at(1));
			Require(IsShape<VectorShape>(argumentShape1) && IsShape<ListShape>(argumentShape2));

			auto listShape = CastShape<ListShape>(argumentShape2);
			auto elementShape = listShape->GetElementShape();
			Require(IsShape<VectorShape>(elementShape));

			auto argumentSize1 = CastShape<VectorShape>(argumentShape1)->GetSize();
			auto argumentSize2 = listShape->GetListSize();
			Require(*argumentSize1 == *argumentSize2);

			return {new TableShape(argumentSize1, CastShape<VectorShape>(elementShape)->GetSize())};
		}
		case HorseIR::BuiltinFunction::Primitive::KeyedTable:
		{
			//TODO:
			break;
		}
		case HorseIR::BuiltinFunction::Primitive::Keys:
		{
			//TODO:
			break;
		}
		case HorseIR::BuiltinFunction::Primitive::Values:
		{
			//TODO:
			break;
		}
		case HorseIR::BuiltinFunction::Primitive::Meta:
		{
			//TODO:
			break;
		}
		case HorseIR::BuiltinFunction::Primitive::Fetch:
		{
			//TODO:
			break;
		}
		case HorseIR::BuiltinFunction::Primitive::ColumnValue:
		{
			auto argumentShape = GetShape(arguments.at(0));
			Require(IsShape<TableShape>(argumentShape));

			auto tableShape = CastShape<TableShape>(argumentShape);
			return {new VectorShape(tableShape->GetRowsSize())};
		}
		case HorseIR::BuiltinFunction::Primitive::LoadTable:
		{
			auto tableName = static_cast<const HorseIR::SymbolLiteral *>(arguments.at(0))->GetValue(0)->GetName();

			auto columns = new Shape::SymbolSize(tableName + ".cols");
			auto rows = new Shape::SymbolSize(tableName + ".rows");
			return {new TableShape(columns, rows)};
		}
		case HorseIR::BuiltinFunction::Primitive::DatetimeAdd:
		{
			//TODO:
			break;
		}
		case HorseIR::BuiltinFunction::Primitive::DatetimeSubtract:
		{
			//TODO:
			break;
		}
		case HorseIR::BuiltinFunction::Primitive::JoinIndex:
		{
			//TODO:
			break;
		}
		case HorseIR::BuiltinFunction::Primitive::Index:
		{
			//TODO:
			break;
		}
		case HorseIR::BuiltinFunction::Primitive::IndexAssignment:
		{
			//TODO:
			break;
		}
		case HorseIR::BuiltinFunction::Primitive::LoadCSV:
		{
			//TODO:
			break;
		}
		case HorseIR::BuiltinFunction::Primitive::Print:
		{
			//TODO:
			break;
		}
		case HorseIR::BuiltinFunction::Primitive::Format:
		{
			//TODO:
			break;
		}
		case HorseIR::BuiltinFunction::Primitive::SubString:
		{
			//TODO:
			break;
		}
		default:
		{
			Utils::Logger::LogError("Shape analysis is not supported for builtin function '" + function->GetName() + "'");
		}
	}

	ShapeError(function, arguments);
}         

[[noreturn]] void ShapeAnalysisHelper::ShapeError(const HorseIR::FunctionDeclaration *method, const std::vector<HorseIR::Operand *>& arguments) const
{
	std::stringstream message;
	message << "Incompatible shapes [";

	bool first = true;
	for (const auto& argument : arguments)
	{
		if (!first)
		{
			message << ", ";
		}
		first = false;
		message << *GetShape(argument);
	}

	message << "] to function '" << method->GetName() << "'";
	Utils::Logger::LogError(message.str());
}

void ShapeAnalysisHelper::Visit(const HorseIR::CastExpression *cast)
{
	// Traverse the expression

	ConstVisitor::Visit(cast);

	// Propagate the shape from the expression to the cast

	SetShapes(cast, GetShapes(cast->GetExpression()));
}

void ShapeAnalysisHelper::Visit(const HorseIR::Identifier *identifier)
{
	SetShape(identifier, m_properties.at(identifier->GetSymbol()));
}

void ShapeAnalysisHelper::Visit(const HorseIR::VectorLiteral *literal)
{
	SetShape(literal, new VectorShape(new Shape::ConstantSize(literal->GetCount())));
}

const Shape *ShapeAnalysisHelper::GetShape(const HorseIR::Operand *operand) const
{
	auto& shapes = m_shapes.at(operand);
	if (shapes.size() > 1)
	{
		Utils::Logger::LogError("Operand has more than one shape.");
	}
	return shapes.at(0);
}

void ShapeAnalysisHelper::SetShape(const HorseIR::Operand *operand, const Shape *shape)
{
	m_shapes[operand] = {shape};
}

const std::vector<const Shape *>& ShapeAnalysisHelper::GetShapes(const HorseIR::Expression *expression) const
{
	return m_shapes.at(expression);
}

void ShapeAnalysisHelper::SetShapes(const HorseIR::Expression *expression, const std::vector<const Shape *>& shapes)
{
	m_shapes[expression] = shapes;
}

}
