#include "Analysis/Shape/ShapeAnalysisHelper.h"

#include "Analysis/Helpers/ValueAnalysisHelper.h"
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
	const auto& returnTypes = function->GetReturnTypes();
	if (returnTypes.size() == 1)
	{
		return {ShapeUtils::ShapeFromType(returnTypes.at(0), m_call)};
	}
	else
	{
		std::vector<const Shape *> returnShapes;
		unsigned int tag = 1;
		for (const auto returnType : function->GetReturnTypes())
		{
			returnShapes.push_back(ShapeUtils::ShapeFromType(returnType, m_call, tag++));
		}
		return returnShapes;
	}
}

void ShapeAnalysisHelper::AddScalarConstraint(const Shape::Size *size)
{
	AddEqualityConstraint(size, new Shape::ConstantSize(1));
}

void ShapeAnalysisHelper::AddBinaryConstraint(const Shape::Size *size1, const Shape::Size *size2)
{

}

void ShapeAnalysisHelper::AddEqualityConstraint(const Shape::Size *size1, const Shape::Size *size2)
{

}

std::vector<const Shape *> ShapeAnalysisHelper::AnalyzeCall(const HorseIR::BuiltinFunction *function, const std::vector<HorseIR::Operand *>& arguments)
{
	switch (function->GetPrimitive())
	{
#define Require(x) if (!(x)) break
#define Unsupported() return {new WildcardShape(m_call)}

		// Unary
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
		
		// Date Unary
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
			Require(ShapeUtils::IsShape<VectorShape>(argumentShape));
			return {argumentShape};
		}

		// Binary
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

		// Date Binary
		case HorseIR::BuiltinFunction::Primitive::DatetimeAdd:
		case HorseIR::BuiltinFunction::Primitive::DatetimeSubtract:
		case HorseIR::BuiltinFunction::Primitive::DatetimeDifference:
		{
			auto argumentShape1 = GetShape(arguments.at(0));
			auto argumentShape2 = GetShape(arguments.at(1));
			Require(ShapeUtils::IsShape<VectorShape>(argumentShape1) && ShapeUtils::IsShape<VectorShape>(argumentShape2));

			auto argumentSize1 = ShapeUtils::GetShape<VectorShape>(argumentShape1)->GetSize();
			auto argumentSize2 = ShapeUtils::GetShape<VectorShape>(argumentShape2)->GetSize();

			const Shape::Size *size = nullptr;

			if (*argumentSize1 == *argumentSize2)
			{
				size = argumentSize1;
			}
			else if (ShapeUtils::IsSize<Shape::ConstantSize>(argumentSize1) && ShapeUtils::GetSize<Shape::ConstantSize>(argumentSize1)->GetValue() == 1)
			{
				size = argumentSize2;
			}
			else if (ShapeUtils::IsSize<Shape::ConstantSize>(argumentSize2) && ShapeUtils::GetSize<Shape::ConstantSize>(argumentSize2)->GetValue() == 1)
			{
				size = argumentSize1;
			}
			else if (ShapeUtils::IsSize<Shape::ConstantSize>(argumentSize1) && ShapeUtils::IsSize<Shape::ConstantSize>(argumentSize1))
			{
				auto constant1 = ShapeUtils::GetSize<Shape::ConstantSize>(argumentSize1)->GetValue();
				auto constant2 = ShapeUtils::GetSize<Shape::ConstantSize>(argumentSize2)->GetValue();

				if (constant1 != constant2)
				{
					Utils::Logger::LogError("Binary function '" + function->GetName() + "' requires vector of same length (or broadcast) [" + std::to_string(constant1) + " != " + std::to_string(constant2) + "]");
				}
				size = new Shape::ConstantSize(constant1);
			}
			else
			{
				AddBinaryConstraint(argumentSize1, argumentSize2);
				size = new Shape::DynamicSize(m_call);
			}
			return {new VectorShape(size)};
		}
		
		// Algebraic Unary
		case HorseIR::BuiltinFunction::Primitive::Unique:
		{
			auto argumentShape = GetShape(arguments.at(0));
			Require(ShapeUtils::IsShape<VectorShape>(argumentShape));
			return {new VectorShape(new Shape::DynamicSize(m_call))};
		}
		case HorseIR::BuiltinFunction::Primitive::Range:
		{
			auto argumentShape = GetShape(arguments.at(0));
			Require(ShapeUtils::IsShape<VectorShape>(argumentShape));

			auto vectorShape = ShapeUtils::GetShape<VectorShape>(argumentShape);
			AddScalarConstraint(vectorShape->GetSize());

			if (ValueAnalysisHelper::IsConstant(arguments.at(0)))
			{
				auto value = ValueAnalysisHelper::GetConstant<std::int64_t>(arguments.at(0));
				return {new VectorShape(new Shape::ConstantSize(value.at(0)))};
			}
			return {new VectorShape(new Shape::DynamicSize(m_call))};
		}
		case HorseIR::BuiltinFunction::Primitive::Factorial:
		case HorseIR::BuiltinFunction::Primitive::Reverse:
		{
			auto argumentShape = GetShape(arguments.at(0));
			Require(ShapeUtils::IsShape<VectorShape>(argumentShape));
			return {argumentShape};
		}
		case HorseIR::BuiltinFunction::Primitive::Random:
		case HorseIR::BuiltinFunction::Primitive::Seed:
		case HorseIR::BuiltinFunction::Primitive::Flip:
		{
			//TODO: Unknown shape rules
			Unsupported();
		}
		case HorseIR::BuiltinFunction::Primitive::Where:
		{
			auto argumentShape = GetShape(arguments.at(0));
			if (ShapeUtils::IsShape<VectorShape>(argumentShape))
			{
				return {new VectorShape(new Shape::DynamicSize(m_call))};
			}
			else if (ShapeUtils::IsShape<ListShape>(argumentShape))
			{
				auto listShape = ShapeUtils::GetShape<ListShape>(argumentShape);
				return {new ListShape(listShape->GetListSize(), {new VectorShape(new Shape::DynamicSize(m_call))})};
			}
			break;
		}
		case HorseIR::BuiltinFunction::Primitive::Group:
		{
			auto argumentShape = GetShape(arguments.at(0));
			if (ShapeUtils::IsShape<VectorShape>(argumentShape))
			{
				return {new DictionaryShape(
						new VectorShape(new Shape::DynamicSize(m_call, 1)),
						new ListShape(new Shape::ConstantSize(1), {new VectorShape(new Shape::DynamicSize(m_call, 2))})
				)};
			}
			else if (ShapeUtils::IsShape<ListShape>(argumentShape))
			{
				auto listShape = ShapeUtils::GetShape<ListShape>(argumentShape);
				return {new DictionaryShape(
						new VectorShape(new Shape::DynamicSize(m_call, 1)),
						new ListShape(listShape->GetListSize(), {new VectorShape(new Shape::DynamicSize(m_call, 2))})
				)};
			}
			break;
		}

		// Algebraic Binary
		case HorseIR::BuiltinFunction::Primitive::Append:
		{
			//TODO: Unknown shape rule
			Unsupported();
		}
		case HorseIR::BuiltinFunction::Primitive::Replicate:
		{
			//TODO: Unknown shape rule
			Unsupported();
		}
		case HorseIR::BuiltinFunction::Primitive::Like:
		{
			auto argumentShape1 = GetShape(arguments.at(0));
			auto argumentShape2 = GetShape(arguments.at(1));
			Require(ShapeUtils::IsShape<VectorShape>(argumentShape1) && ShapeUtils::IsShape<VectorShape>(argumentShape2));

			auto vectorShape2 = ShapeUtils::GetShape<VectorShape>(argumentShape2);
			AddScalarConstraint(vectorShape2->GetSize());

			return {argumentShape1};
		}
		case HorseIR::BuiltinFunction::Primitive::Compress:
		{
			auto argumentShape1 = GetShape(arguments.at(0));
			auto argumentShape2 = GetShape(arguments.at(1));
			Require(ShapeUtils::IsShape<VectorShape>(argumentShape1) && ShapeUtils::IsShape<VectorShape>(argumentShape2));

			auto argumentSize1 = ShapeUtils::GetShape<VectorShape>(argumentShape1)->GetSize();
			auto argumentSize2 = ShapeUtils::GetShape<VectorShape>(argumentShape2)->GetSize();

			//TODO: Constraint

			auto dynamic1 = ShapeUtils::IsSize<Shape::DynamicSize>(argumentSize1);
			auto dynamic2 = ShapeUtils::IsSize<Shape::DynamicSize>(argumentSize2);
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
			//TODO: Unknown shape rule
			Unsupported();
		}
		case HorseIR::BuiltinFunction::Primitive::IndexOf:
		{
			auto argumentShape1 = GetShape(arguments.at(0));
			auto argumentShape2 = GetShape(arguments.at(1));
			Require(ShapeUtils::IsShape<VectorShape>(argumentShape1) && ShapeUtils::IsShape<VectorShape>(argumentShape2));
			return {argumentShape1};
		}
		case HorseIR::BuiltinFunction::Primitive::Take:
		case HorseIR::BuiltinFunction::Primitive::Drop:
		{
			//TODO: Unknown shape rule
			Unsupported();
		}
		case HorseIR::BuiltinFunction::Primitive::Order:
		{
			auto argumentShape1 = GetShape(arguments.at(0));
			auto argumentShape2 = GetShape(arguments.at(1));
			Require(ShapeUtils::IsShape<VectorShape>(argumentShape2));

			auto vectorShape2 = ShapeUtils::GetShape<VectorShape>(argumentShape2);
			if (ShapeUtils::IsShape<VectorShape>(argumentShape1))
			{
				AddScalarConstraint(vectorShape2->GetSize());
				return {argumentShape1};
			}
			else if (ShapeUtils::IsShape<ListShape>(argumentShape1))
			{
				auto elementShapes = ShapeUtils::GetShape<ListShape>(argumentShape1)->GetElementShapes();
				Require(elementShapes.size() == 1);

				auto elementShape = elementShapes.at(0);
				Require(ShapeUtils::IsShape<VectorShape>(elementShape));

				auto elementVectorShape = ShapeUtils::GetShape<VectorShape>(elementShape);
				AddEqualityConstraint(vectorShape2->GetSize(), elementVectorShape->GetSize());

				return {elementShape};
			}
			break;
		}
		case HorseIR::BuiltinFunction::Primitive::Member:
		{
			auto argumentShape1 = GetShape(arguments.at(0));
			auto argumentShape2 = GetShape(arguments.at(1));
			Require(ShapeUtils::IsShape<VectorShape>(argumentShape1) && ShapeUtils::IsShape<VectorShape>(argumentShape2));
			return {argumentShape1};
		}
		case HorseIR::BuiltinFunction::Primitive::Vector:
		{
			auto argumentShape1 = GetShape(arguments.at(0));
			auto argumentShape2 = GetShape(arguments.at(1));
			Require(ShapeUtils::IsShape<VectorShape>(argumentShape1) && ShapeUtils::IsShape<VectorShape>(argumentShape2));

			auto vectorShape1 = ShapeUtils::GetShape<VectorShape>(argumentShape1);
			auto vectorShape2 = ShapeUtils::GetShape<VectorShape>(argumentShape2);
			AddScalarConstraint(vectorShape1->GetSize());
			AddScalarConstraint(vectorShape2->GetSize());

			if (ValueAnalysisHelper::IsConstant(arguments.at(0)))
			{
				auto value = ValueAnalysisHelper::GetConstant<std::int64_t>(arguments.at(0));
				return {new VectorShape(new Shape::ConstantSize(value.at(0)))};
			}
			return {new VectorShape(new Shape::DynamicSize(m_call))};
		}

		// Reduction

		// @count and @len are aliases
		case HorseIR::BuiltinFunction::Primitive::Length:
		case HorseIR::BuiltinFunction::Primitive::Count:
		case HorseIR::BuiltinFunction::Primitive::Sum:
		case HorseIR::BuiltinFunction::Primitive::Average:
		case HorseIR::BuiltinFunction::Primitive::Minimum:
		case HorseIR::BuiltinFunction::Primitive::Maximum:
		{
			auto argumentShape = GetShape(arguments.at(0));
			Require(ShapeUtils::IsShape<VectorShape>(argumentShape));
			return {new VectorShape(new Shape::ConstantSize(1))};
		}

		// List
		case HorseIR::BuiltinFunction::Primitive::Raze:
		{
			//TODO: Unknown shape rule
			Unsupported();
		}
		case HorseIR::BuiltinFunction::Primitive::List:
		{
			//TODO: Update shape rule
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
					shape = new WildcardShape(m_call);
					break;
				}
			}
			return {new ListShape(new Shape::ConstantSize(arguments.size()), {shape})};
		}
		case HorseIR::BuiltinFunction::Primitive::ToList:
		{
			Unsupported();
		}
		case HorseIR::BuiltinFunction::Primitive::Each:
		{
			Unsupported();
		}
		case HorseIR::BuiltinFunction::Primitive::EachItem:
		{
			Unsupported();
		}
		case HorseIR::BuiltinFunction::Primitive::EachLeft:
		{
			Unsupported();
		}
		case HorseIR::BuiltinFunction::Primitive::EachRight:
		{
			Unsupported();
		}
		case HorseIR::BuiltinFunction::Primitive::Match:
		{
			Unsupported();
		}

		// Database
		case HorseIR::BuiltinFunction::Primitive::Enum:
		{
			Unsupported();
		}
		case HorseIR::BuiltinFunction::Primitive::Dictionary:
		{
			auto argumentShape1 = GetShape(arguments.at(0));
			auto argumentShape2 = GetShape(arguments.at(1));
			//TODO: Constraint
			return {new DictionaryShape(argumentShape1, argumentShape2)};
		}
		case HorseIR::BuiltinFunction::Primitive::Table:
		{
			//TODO: Update shape rule
			auto argumentShape1 = GetShape(arguments.at(0));
			auto argumentShape2 = GetShape(arguments.at(1));
			Require(ShapeUtils::IsShape<VectorShape>(argumentShape1) && ShapeUtils::IsShape<ListShape>(argumentShape2));

			auto listShape = ShapeUtils::GetShape<ListShape>(argumentShape2);
			auto elementShapes = listShape->GetElementShapes();
			Require(elementShapes.size() == 1);

			auto elementShape = elementShapes.at(0);
			Require(ShapeUtils::IsShape<VectorShape>(elementShape));

			auto argumentSize1 = ShapeUtils::GetShape<VectorShape>(argumentShape1)->GetSize();
			auto argumentSize2 = listShape->GetListSize();
			Require(*argumentSize1 == *argumentSize2);

			return {new TableShape(argumentSize1, ShapeUtils::GetShape<VectorShape>(elementShape)->GetSize())};
		}
		case HorseIR::BuiltinFunction::Primitive::KeyedTable:
		{
			auto argumentShape1 = GetShape(arguments.at(0));
			auto argumentShape2 = GetShape(arguments.at(1));
			Require(ShapeUtils::IsShape<TableShape>(argumentShape1) && ShapeUtils::IsShape<TableShape>(argumentShape2));
			return {new KeyedTableShape(ShapeUtils::GetShape<TableShape>(argumentShape1), ShapeUtils::GetShape<TableShape>(argumentShape2))};
		}
		case HorseIR::BuiltinFunction::Primitive::Keys:
		{
			auto argumentShape = GetShape(arguments.at(0));
			if (ShapeUtils::IsShape<DictionaryShape>(argumentShape))
			{
				return {ShapeUtils::GetShape<DictionaryShape>(argumentShape)->GetKeyShape()};
			}
			else if (ShapeUtils::IsShape<TableShape>(argumentShape))
			{
				return {new VectorShape(ShapeUtils::GetShape<TableShape>(argumentShape)->GetColumnsSize())};
			}
			else if (ShapeUtils::IsShape<KeyedTableShape>(argumentShape))
			{
				return {ShapeUtils::GetShape<KeyedTableShape>(argumentShape)->GetKeyShape()};
			}
			else if (ShapeUtils::IsShape<EnumerationShape>(argumentShape))
			{
				return {new VectorShape(ShapeUtils::GetShape<EnumerationShape>(argumentShape)->GetMapSize())};
			}
			break;
		}
		case HorseIR::BuiltinFunction::Primitive::Values:
		{
			auto argumentShape = GetShape(arguments.at(0));
			if (ShapeUtils::IsShape<DictionaryShape>(argumentShape))
			{
				return {ShapeUtils::GetShape<DictionaryShape>(argumentShape)->GetValueShape()};
			}
			else if (ShapeUtils::IsShape<TableShape>(argumentShape))
			{
				auto tableShape = ShapeUtils::GetShape<TableShape>(argumentShape);
				auto colsSize = tableShape->GetColumnsSize();
				auto rowsSize = tableShape->GetRowsSize();
				return {new ListShape(colsSize, {new VectorShape(rowsSize)})};
			}
			else if (ShapeUtils::IsShape<KeyedTableShape>(argumentShape))
			{
				return {ShapeUtils::GetShape<KeyedTableShape>(argumentShape)->GetValueShape()};
			}
			else if (ShapeUtils::IsShape<EnumerationShape>(argumentShape))
			{
				return {new VectorShape(ShapeUtils::GetShape<EnumerationShape>(argumentShape)->GetMapSize())};
			}
			break;
		}
		case HorseIR::BuiltinFunction::Primitive::Meta:
		{
			auto argumentShape = GetShape(arguments.at(0));
			Require(ShapeUtils::IsShape<TableShape>(argumentShape) || ShapeUtils::IsShape<KeyedTableShape>(argumentShape));
			return {new TableShape(new Shape::DynamicSize(m_call, 1), new Shape::DynamicSize(m_call, 2))};
		}
		case HorseIR::BuiltinFunction::Primitive::Fetch:
		{
			auto argumentShape = GetShape(arguments.at(0));
			Require(ShapeUtils::IsShape<EnumerationShape>(argumentShape));
			return {new VectorShape(ShapeUtils::GetShape<EnumerationShape>(argumentShape)->GetMapSize())};
		}
		case HorseIR::BuiltinFunction::Primitive::ColumnValue:
		{
			auto argumentShape = GetShape(arguments.at(0));
			Require(ShapeUtils::IsShape<TableShape>(argumentShape));
			auto tableShape = ShapeUtils::GetShape<TableShape>(argumentShape);
			return {new VectorShape(tableShape->GetRowsSize())};
		}
		case HorseIR::BuiltinFunction::Primitive::LoadTable:
		{
			//TODO: need to be more careful here with casting
			auto tableName = static_cast<const HorseIR::SymbolLiteral *>(arguments.at(0))->GetValue(0)->GetName();
			auto columns = new Shape::SymbolSize(tableName + ".cols");
			auto rows = new Shape::SymbolSize(tableName + ".rows");
			return {new TableShape(columns, rows)};
		}
		case HorseIR::BuiltinFunction::Primitive::JoinIndex:
		{
			//TODO: Unknown shape rule
			Unsupported();
		}

		// Indexing
		case HorseIR::BuiltinFunction::Primitive::Index:
		{
			auto argumentShape1 = GetShape(arguments.at(0));
			auto argumentShape2 = GetShape(arguments.at(1));
			Require(ShapeUtils::IsShape<VectorShape>(argumentShape1) && ShapeUtils::IsShape<VectorShape>(argumentShape2));
			return {argumentShape2};
		}
		case HorseIR::BuiltinFunction::Primitive::IndexAssignment:
		{
			auto argumentShape1 = GetShape(arguments.at(0));
			auto argumentShape2 = GetShape(arguments.at(1));
			auto argumentShape3 = GetShape(arguments.at(2));
			Require(ShapeUtils::IsShape<VectorShape>(argumentShape1) && ShapeUtils::IsShape<VectorShape>(argumentShape2) && ShapeUtils::IsShape<VectorShape>(argumentShape3));

			auto vectorShape2 = ShapeUtils::GetShape<VectorShape>(argumentShape2);
			auto vectorShape3 = ShapeUtils::GetShape<VectorShape>(argumentShape3);
			AddEqualityConstraint(vectorShape2->GetSize(), vectorShape3->GetSize());
			return {argumentShape1};
		}

		// Other
		case HorseIR::BuiltinFunction::Primitive::LoadCSV:
		{
			auto argumentShape = GetShape(arguments.at(0));
			Require(ShapeUtils::IsShape<VectorShape>(argumentShape));
			return {new TableShape(new Shape::DynamicSize(m_call, 1), new Shape::DynamicSize(m_call, 2))};
		}
		case HorseIR::BuiltinFunction::Primitive::Print:
		case HorseIR::BuiltinFunction::Primitive::Format:
		case HorseIR::BuiltinFunction::Primitive::String:
		{
			return {new VectorShape(new Shape::ConstantSize(1))};
		}
		case HorseIR::BuiltinFunction::Primitive::SubString:
		{
			auto argumentShape = GetShape(arguments.at(2));
			Require(ShapeUtils::IsShape<VectorShape>(argumentShape));
			return {argumentShape};
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

	cast->GetExpression()->Accept(*this);

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
