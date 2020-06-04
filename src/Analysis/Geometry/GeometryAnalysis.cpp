#include "Analysis/Geometry/GeometryAnalysis.h"

#include "Analysis/Shape/ShapeCollector.h"
#include "Analysis/Shape/ShapeUtils.h"

#include "HorseIR/Analysis/StatementAnalysisPrinter.h"

#include "Utils/Chrono.h"
#include "Utils/Logger.h"
#include "Utils/Options.h"

namespace Analysis {

void GeometryAnalysis::Analyze(const HorseIR::Function *function)
{
	auto timeGeometry_start = Utils::Chrono::Start("Geometry analysis '" + function->GetName() + "'");
	function->Accept(*this);
	Utils::Chrono::End(timeGeometry_start);

	if (Utils::Options::Present(Utils::Options::Opt_Print_analysis))
	{
		Utils::Logger::LogInfo("Geometry analysis '" + function->GetName() + "'");

		auto string = HorseIR::StatementAnalysisPrinter::PrettyString(*this, function);
		Utils::Logger::LogInfo(string, 0, true, Utils::Logger::NoPrefix);
	}
}

bool GeometryAnalysis::VisitIn(const HorseIR::DeclarationStatement *declarationS)
{
	const auto& outShapes = m_shapeAnalysis.GetOutSet(declarationS);
	m_geometries[declarationS] = outShapes.first.at(declarationS->GetDeclaration()->GetSymbol());
	return false;
}

bool GeometryAnalysis::VisitIn(const HorseIR::Statement *statement)
{
	m_currentStatement = statement;
	return true;
}

void GeometryAnalysis::VisitOut(const HorseIR::Statement *statement)
{
	if (m_currentGeometry != nullptr)
	{
		m_geometries[statement] = m_currentGeometry;
	}

	m_currentStatement = nullptr;
	m_currentGeometry = nullptr;
}

bool GeometryAnalysis::VisitIn(const HorseIR::IfStatement *ifS)
{
	m_currentStatement = ifS;
	ifS->GetCondition()->Accept(*this);
	if (m_currentGeometry != nullptr)
	{
		m_geometries[ifS] = m_currentGeometry;
	}
	ifS->GetTrueBlock()->Accept(*this);
	if (ifS->HasElseBranch())
	{
		ifS->GetElseBlock()->Accept(*this);
	}
	return false;
}

bool GeometryAnalysis::VisitIn(const HorseIR::WhileStatement *whileS)
{
	m_currentStatement = whileS;
	whileS->GetCondition()->Accept(*this);
	if (m_currentGeometry != nullptr)
	{
		m_geometries[whileS] = m_currentGeometry;
	}
	whileS->GetBody()->Accept(*this);
	return false;
}

bool GeometryAnalysis::VisitIn(const HorseIR::RepeatStatement *repeatS)
{
	m_currentStatement = repeatS;
	repeatS->GetCondition()->Accept(*this);
	if (m_currentGeometry != nullptr)
	{
		m_geometries[repeatS] = m_currentGeometry;
	}
	repeatS->GetBody()->Accept(*this);
	return false;
}

bool GeometryAnalysis::VisitIn(const HorseIR::AssignStatement *assignS)
{
	// Traverse the RHS of the assignment for operating geometry

	m_currentStatement = assignS;
	assignS->GetExpression()->Accept(*this);
	return false;
}

bool GeometryAnalysis::VisitIn(const HorseIR::CallExpression *call)
{
	// Analyze the call geometry, ignore parameters

	m_call = call;
	m_currentGeometry = AnalyzeCall(call->GetFunctionLiteral()->GetFunction(), call->GetArguments());
	m_call = nullptr;

	return false;
}

const Shape *GeometryAnalysis::AnalyzeCall(const HorseIR::FunctionDeclaration *function, const std::vector<HorseIR::Operand *>& arguments)
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

const Shape *GeometryAnalysis::AnalyzeCall(const HorseIR::Function *function, const std::vector<HorseIR::Operand *>& arguments)
{
	// Without interprocedural analysis, assume the function operates on an unknown geometry

	return new WildcardShape();
}

const Shape *GeometryAnalysis::AnalyzeCall(const HorseIR::BuiltinFunction *function, const std::vector<HorseIR::Operand *>& arguments)
{
	const auto& inShapes = m_shapeAnalysis.GetInSet(m_currentStatement);
	const auto& outShapes = m_shapeAnalysis.GetOutSet(m_currentStatement);

	switch (function->GetPrimitive())
	{
#define Require(x) if (!(x)) break
#define CPU() return {new WildcardShape()}
#define Independent() return {new WildcardShape()}

		// ---------------
		// Vector Output Geometry
		// ---------------

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
		
		// Algebraic Unary
		case HorseIR::BuiltinFunction::Primitive::Range:
		case HorseIR::BuiltinFunction::Primitive::Factorial:
		case HorseIR::BuiltinFunction::Primitive::Reverse:

		// Algebraic Binary
		case HorseIR::BuiltinFunction::Primitive::Random_k:
		case HorseIR::BuiltinFunction::Primitive::IndexOf:
		case HorseIR::BuiltinFunction::Primitive::Member:
		case HorseIR::BuiltinFunction::Primitive::Vector:

		// Indexing
		case HorseIR::BuiltinFunction::Primitive::Index:
		{
			const auto& shapes = m_shapeAnalysis.GetShapes(m_call);
			Require(shapes.size() == 1);
			return shapes.at(0);
		}

		// --------------------
		// Vector Input Geometry
		// --------------------

		// Indexing
		case HorseIR::BuiltinFunction::Primitive::IndexAssignment:
		{
			// We operate on the size of the index set

			return ShapeCollector::ShapeFromOperand(inShapes, arguments.at(1));
		}

		// Algebraic Unary
		case HorseIR::BuiltinFunction::Primitive::Take:
		case HorseIR::BuiltinFunction::Primitive::Drop:
		{
			// We operate on the size of the input vector

			return ShapeCollector::ShapeFromOperand(inShapes, arguments.at(1));
		}

		// Algebraic Binary
		case HorseIR::BuiltinFunction::Primitive::Append:
		{
			const auto& shapes = m_shapeAnalysis.GetShapes(m_call);
			Require(shapes.size() == 1);

			// Only support vector appending on the GPU

			if (const auto vectorShape = ShapeUtils::GetShape<VectorShape>(shapes.at(0)))
			{
				return vectorShape;
			}
			CPU();
		}

		// --------------------
		// Compression Geometry
		// --------------------

		// Algebraic Unary
		case HorseIR::BuiltinFunction::Primitive::Where:

		// Algebraic Binary
		case HorseIR::BuiltinFunction::Primitive::Compress:
		{
			const auto& shapes = m_shapeAnalysis.GetShapes(m_call);
			Require(shapes.size() == 1 && ShapeUtils::IsShape<VectorShape>(shapes.at(0)));

			// Recover the full size from the compression

			auto vectorShape = ShapeUtils::GetShape<VectorShape>(shapes.at(0));
			Require(ShapeUtils::IsSize<Shape::CompressedSize>(vectorShape->GetSize()));

			auto fullSize = ShapeUtils::GetSize<Shape::CompressedSize>(vectorShape->GetSize())->GetSize();
			return new VectorShape(fullSize);
		}

		// ----------------------
		// Independent Operations
		// ----------------------

		// Algebraic Unary
		case HorseIR::BuiltinFunction::Primitive::Unique:
		case HorseIR::BuiltinFunction::Primitive::Group:

		// Algebriac Binary
		case HorseIR::BuiltinFunction::Primitive::Order:

		// Database
		case HorseIR::BuiltinFunction::Primitive::JoinIndex:
		{
			Independent();
		}

		// --------------------
		// Reduction Operations
		// --------------------

		// Reduction
		case HorseIR::BuiltinFunction::Primitive::Length:
		case HorseIR::BuiltinFunction::Primitive::Sum:
		case HorseIR::BuiltinFunction::Primitive::Average:
		case HorseIR::BuiltinFunction::Primitive::Minimum:
		case HorseIR::BuiltinFunction::Primitive::Maximum:
		{
			return ShapeCollector::ShapeFromOperand(inShapes, arguments.at(0));
		}

		// ---------------
		// List Operations
		// ---------------

		//TODO: Use the nested function for geometry computation

		// List
		case HorseIR::BuiltinFunction::Primitive::Each:
		case HorseIR::BuiltinFunction::Primitive::EachItem:
		{
			return ShapeCollector::ShapeFromOperand(inShapes, arguments.at(1));
		}
		case HorseIR::BuiltinFunction::Primitive::EachLeft:
		{
			return ShapeCollector::ShapeFromOperand(inShapes, arguments.at(1));
		}
		case HorseIR::BuiltinFunction::Primitive::EachRight:
		{
			return ShapeCollector::ShapeFromOperand(inShapes, arguments.at(2));
		}

		// ---------------
		// Cell Operations
		// ---------------

		// List
		case HorseIR::BuiltinFunction::Primitive::Raze:
		{
			return ShapeCollector::ShapeFromOperand(inShapes, arguments.at(0));
		}
		case HorseIR::BuiltinFunction::Primitive::ToList:
		{
			const auto& shapes = m_shapeAnalysis.GetShapes(m_call);
			Require(shapes.size() == 1);
			return shapes.at(0);
		}
		
		// --------------
		// CPU Operations
		// --------------

		// Algebraic Unary
		case HorseIR::BuiltinFunction::Primitive::Random:
		case HorseIR::BuiltinFunction::Primitive::Seed:
		case HorseIR::BuiltinFunction::Primitive::Flip:

		// Algebraic Binary
		case HorseIR::BuiltinFunction::Primitive::Replicate:
		case HorseIR::BuiltinFunction::Primitive::Like:

		// List
		case HorseIR::BuiltinFunction::Primitive::List:
		case HorseIR::BuiltinFunction::Primitive::Match:

		// Database
		case HorseIR::BuiltinFunction::Primitive::Enum:
		case HorseIR::BuiltinFunction::Primitive::Dictionary:
		case HorseIR::BuiltinFunction::Primitive::Table:
		case HorseIR::BuiltinFunction::Primitive::KeyedTable:
		case HorseIR::BuiltinFunction::Primitive::Keys:
		case HorseIR::BuiltinFunction::Primitive::Values:
		case HorseIR::BuiltinFunction::Primitive::Meta:
		case HorseIR::BuiltinFunction::Primitive::Fetch:
		case HorseIR::BuiltinFunction::Primitive::ColumnValue:
		case HorseIR::BuiltinFunction::Primitive::LoadTable:

		// Other
		case HorseIR::BuiltinFunction::Primitive::LoadCSV:
		case HorseIR::BuiltinFunction::Primitive::Print:
		case HorseIR::BuiltinFunction::Primitive::String:
		case HorseIR::BuiltinFunction::Primitive::SubString:
		{
			CPU();
		}

		// GPU
		case HorseIR::BuiltinFunction::Primitive::GPUOrderLib:
		case HorseIR::BuiltinFunction::Primitive::GPUGroupLib:
		case HorseIR::BuiltinFunction::Primitive::GPUUniqueLib:
		case HorseIR::BuiltinFunction::Primitive::GPULoopJoinLib:
		case HorseIR::BuiltinFunction::Primitive::GPUHashJoinLib:
		{
			CPU();
		}
		case HorseIR::BuiltinFunction::Primitive::GPUOrderInit:
		{
			const auto& shapes = m_shapeAnalysis.GetShapes(m_call);
			auto indexShape = shapes.at(0);
			Require(ShapeUtils::IsShape<VectorShape>(indexShape));
			return indexShape;
		}
		case HorseIR::BuiltinFunction::Primitive::GPUOrder:
		case HorseIR::BuiltinFunction::Primitive::GPUOrderShared:
		{
			auto indexShape = ShapeCollector::ShapeFromOperand(inShapes, arguments.at(0));
			Require(ShapeUtils::IsShape<VectorShape>(indexShape));

			// Divide by 2 if constant, otherwise return a new dynamic size

			auto vectorShape = ShapeUtils::GetShape<VectorShape>(indexShape);
			if (const auto constantSize = ShapeUtils::GetSize<Shape::ConstantSize>(vectorShape->GetSize()))
			{
				return new VectorShape(new Shape::ConstantSize(constantSize->GetValue() / 2));
			}
			return new VectorShape(new Shape::DynamicSize(m_call));
		}
		case HorseIR::BuiltinFunction::Primitive::GPUGroup:
		case HorseIR::BuiltinFunction::Primitive::GPUUnique:
		{
			auto indexShape = ShapeCollector::ShapeFromOperand(inShapes, arguments.at(0));
			Require(ShapeUtils::IsShape<VectorShape>(indexShape));
			return indexShape;
		}
		case HorseIR::BuiltinFunction::Primitive::GPULoopJoinCount:
		case HorseIR::BuiltinFunction::Primitive::GPULoopJoin:
		case HorseIR::BuiltinFunction::Primitive::GPUHashJoinCount:
		case HorseIR::BuiltinFunction::Primitive::GPUHashJoin:
		{
			// Left data geometry

			auto leftIndex = arguments.size();
			switch (function->GetPrimitive())
			{
				case HorseIR::BuiltinFunction::Primitive::GPULoopJoinCount:
				{
					leftIndex -= 2;
					break;
				}
				case HorseIR::BuiltinFunction::Primitive::GPULoopJoin:
				{
					leftIndex -= 4;
					break;
				}
				case HorseIR::BuiltinFunction::Primitive::GPUHashJoinCount:
				{
					leftIndex -= 3;
					break;
				}
				case HorseIR::BuiltinFunction::Primitive::GPUHashJoin:
				{
					leftIndex -= 5;
					break;
				}
			}
			auto leftShape = ShapeCollector::ShapeFromOperand(inShapes, arguments.at(leftIndex));
			if (const auto vectorShape = ShapeUtils::GetShape<VectorShape>(leftShape))
			{
				return vectorShape;
			}
			else if (const auto listShape = ShapeUtils::GetShape<ListShape>(leftShape))
			{
				auto cellShape = ShapeUtils::MergeShapes(listShape->GetElementShapes());
				Require(ShapeUtils::IsShape<VectorShape>(cellShape));
				return cellShape;
			}
			break;
		}
		case HorseIR::BuiltinFunction::Primitive::GPUHashCreate:
		{
			auto dataShape = ShapeCollector::ShapeFromOperand(inShapes, arguments.at(0));
			if (const auto vectorShape = ShapeUtils::GetShape<VectorShape>(dataShape))
			{
				return vectorShape;
			}
			else if (const auto listShape = ShapeUtils::GetShape<ListShape>(dataShape))
			{
				auto cellShape = ShapeUtils::MergeShapes(listShape->GetElementShapes());
				Require(ShapeUtils::IsShape<VectorShape>(cellShape));
				return cellShape;
			}
			break;
		}
		default:
		{
			Utils::Logger::LogError("Geometry analysis is not supported for builtin function '" + function->GetName() + "'");
		}
	}

	Utils::Logger::LogError("Geometry analysis recieved unexpected shapes");
}

bool GeometryAnalysis::VisitIn(const HorseIR::Operand *operand)
{
	const auto& inShapes = m_shapeAnalysis.GetInSet(m_currentStatement);
	m_currentGeometry = ShapeCollector::ShapeFromOperand(inShapes, operand);
	return false;
}

std::string GeometryAnalysis::DebugString(const HorseIR::Statement *statement, unsigned int indent) const
{
	std::stringstream string;
	for (auto i = 0u; i < indent; ++i)
	{
		string << "\t";
	}
	string << *m_geometries.at(statement);
	return string.str();
}

}
