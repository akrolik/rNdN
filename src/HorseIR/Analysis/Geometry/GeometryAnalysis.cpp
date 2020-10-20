#include "HorseIR/Analysis/Geometry/GeometryAnalysis.h"

#include "HorseIR/Analysis/Shape/ShapeCollector.h"
#include "HorseIR/Analysis/Shape/ShapeUtils.h"

#include "HorseIR/Analysis/Framework/StatementAnalysisPrinter.h"

#include "Utils/Chrono.h"
#include "Utils/Logger.h"
#include "Utils/Options.h"

namespace HorseIR {
namespace Analysis {

void GeometryAnalysis::Analyze(const Function *function)
{
	auto timeGeometry_start = Utils::Chrono::Start("Geometry analysis '" + function->GetName() + "'");
	function->Accept(*this);
	Utils::Chrono::End(timeGeometry_start);

	if (Utils::Options::Present(Utils::Options::Opt_Print_analysis))
	{
		Utils::Logger::LogInfo("Geometry analysis '" + function->GetName() + "'");

		auto string = StatementAnalysisPrinter::PrettyString(*this, function);
		Utils::Logger::LogInfo(string, 0, true, Utils::Logger::NoPrefix);
	}
}

bool GeometryAnalysis::VisitIn(const DeclarationStatement *declarationS)
{
	const auto& outShapes = m_shapeAnalysis.GetOutSet(declarationS);
	m_geometries[declarationS] = outShapes.first.at(declarationS->GetDeclaration()->GetSymbol());
	return false;
}

bool GeometryAnalysis::VisitIn(const Statement *statement)
{
	m_currentStatement = statement;
	return true;
}

void GeometryAnalysis::VisitOut(const Statement *statement)
{
	if (m_currentGeometry != nullptr)
	{
		m_geometries[statement] = m_currentGeometry;
	}

	m_currentStatement = nullptr;
	m_currentGeometry = nullptr;
}

bool GeometryAnalysis::VisitIn(const IfStatement *ifS)
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

bool GeometryAnalysis::VisitIn(const WhileStatement *whileS)
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

bool GeometryAnalysis::VisitIn(const RepeatStatement *repeatS)
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

bool GeometryAnalysis::VisitIn(const AssignStatement *assignS)
{
	// Traverse the RHS of the assignment for operating geometry

	m_currentStatement = assignS;
	assignS->GetExpression()->Accept(*this);
	return false;
}

bool GeometryAnalysis::VisitIn(const CallExpression *call)
{
	// Analyze the call geometry, ignore parameters

	m_call = call;
	m_currentGeometry = AnalyzeCall(call->GetFunctionLiteral()->GetFunction(), call->GetArguments());
	m_call = nullptr;

	return false;
}

const Shape *GeometryAnalysis::AnalyzeCall(const FunctionDeclaration *function, const std::vector<Operand *>& arguments)
{
	switch (function->GetKind())
	{
		case FunctionDeclaration::Kind::Builtin:
			return AnalyzeCall(static_cast<const BuiltinFunction *>(function), arguments);
		case FunctionDeclaration::Kind::Definition:
			return AnalyzeCall(static_cast<const Function *>(function), arguments);
		default:
			Utils::Logger::LogError("Unsupported function kind");
	}
}

const Shape *GeometryAnalysis::AnalyzeCall(const Function *function, const std::vector<Operand *>& arguments)
{
	// Without interprocedural analysis, assume the function operates on an unknown geometry

	return new WildcardShape();
}

const Shape *GeometryAnalysis::AnalyzeCall(const BuiltinFunction *function, const std::vector<Operand *>& arguments)
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
		case BuiltinFunction::Primitive::Absolute:
		case BuiltinFunction::Primitive::Negate:
		case BuiltinFunction::Primitive::Ceiling:
		case BuiltinFunction::Primitive::Floor:
		case BuiltinFunction::Primitive::Round:
		case BuiltinFunction::Primitive::Conjugate:
		case BuiltinFunction::Primitive::Reciprocal:
		case BuiltinFunction::Primitive::Sign:
		case BuiltinFunction::Primitive::Pi:
		case BuiltinFunction::Primitive::Not:
		case BuiltinFunction::Primitive::Logarithm:
		case BuiltinFunction::Primitive::Logarithm2:
		case BuiltinFunction::Primitive::Logarithm10:
		case BuiltinFunction::Primitive::SquareRoot:
		case BuiltinFunction::Primitive::Exponential:
		case BuiltinFunction::Primitive::Cosine:
		case BuiltinFunction::Primitive::Sine:
		case BuiltinFunction::Primitive::Tangent:
		case BuiltinFunction::Primitive::InverseCosine:
		case BuiltinFunction::Primitive::InverseSine:
		case BuiltinFunction::Primitive::InverseTangent:
		case BuiltinFunction::Primitive::HyperbolicCosine:
		case BuiltinFunction::Primitive::HyperbolicSine:
		case BuiltinFunction::Primitive::HyperbolicTangent:
		case BuiltinFunction::Primitive::HyperbolicInverseCosine:
		case BuiltinFunction::Primitive::HyperbolicInverseSine:
		case BuiltinFunction::Primitive::HyperbolicInverseTangent:
		
		// Date Unary
		case BuiltinFunction::Primitive::Date:
		case BuiltinFunction::Primitive::DateYear:
		case BuiltinFunction::Primitive::DateMonth:
		case BuiltinFunction::Primitive::DateDay:
		case BuiltinFunction::Primitive::Time:
		case BuiltinFunction::Primitive::TimeHour:
		case BuiltinFunction::Primitive::TimeMinute:
		case BuiltinFunction::Primitive::TimeSecond:
		case BuiltinFunction::Primitive::TimeMillisecond:

		// Binary
		case BuiltinFunction::Primitive::Less:
		case BuiltinFunction::Primitive::Greater:
		case BuiltinFunction::Primitive::LessEqual:
		case BuiltinFunction::Primitive::GreaterEqual:
		case BuiltinFunction::Primitive::Equal:
		case BuiltinFunction::Primitive::NotEqual:
		case BuiltinFunction::Primitive::Plus:
		case BuiltinFunction::Primitive::Minus:
		case BuiltinFunction::Primitive::Multiply:
		case BuiltinFunction::Primitive::Divide:
		case BuiltinFunction::Primitive::Power:
		case BuiltinFunction::Primitive::LogarithmBase:
		case BuiltinFunction::Primitive::Modulo:
		case BuiltinFunction::Primitive::And:
		case BuiltinFunction::Primitive::Or:
		case BuiltinFunction::Primitive::Nand:
		case BuiltinFunction::Primitive::Nor:
		case BuiltinFunction::Primitive::Xor:

		// Date Binary
		case BuiltinFunction::Primitive::DatetimeAdd:
		case BuiltinFunction::Primitive::DatetimeSubtract:
		case BuiltinFunction::Primitive::DatetimeDifference:
		
		// Algebraic Unary
		case BuiltinFunction::Primitive::Range:
		case BuiltinFunction::Primitive::Factorial:
		case BuiltinFunction::Primitive::Reverse:

		// Algebraic Binary
		case BuiltinFunction::Primitive::Random_k:
		case BuiltinFunction::Primitive::IndexOf:
		case BuiltinFunction::Primitive::Member:
		case BuiltinFunction::Primitive::Vector:

		// Indexing
		case BuiltinFunction::Primitive::Index:
		{
			const auto& shapes = m_shapeAnalysis.GetShapes(m_call);
			Require(shapes.size() == 1);
			return shapes.at(0);
		}

		// --------------------
		// Vector Input Geometry
		// --------------------

		// Indexing
		case BuiltinFunction::Primitive::IndexAssignment:
		{
			// We operate on the size of the index set

			return ShapeCollector::ShapeFromOperand(inShapes, arguments.at(1));
		}

		// Algebraic Unary
		case BuiltinFunction::Primitive::Unique:
		{
			return ShapeCollector::ShapeFromOperand(inShapes, arguments.at(0));
		}
		case BuiltinFunction::Primitive::Take:
		case BuiltinFunction::Primitive::Drop:
		{
			// We operate on the size of the input vector

			return ShapeCollector::ShapeFromOperand(inShapes, arguments.at(1));
		}

		// Algebraic Binary
		case BuiltinFunction::Primitive::Append:
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
		case BuiltinFunction::Primitive::Where:

		// Algebraic Binary
		case BuiltinFunction::Primitive::Compress:
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
		case BuiltinFunction::Primitive::Group:

		// Algebriac Binary
		case BuiltinFunction::Primitive::Order:

		// Database
		case BuiltinFunction::Primitive::JoinIndex:
		{
			Independent();
		}

		// --------------------
		// Reduction Operations
		// --------------------

		// Reduction
		case BuiltinFunction::Primitive::Length:
		case BuiltinFunction::Primitive::Sum:
		case BuiltinFunction::Primitive::Average:
		case BuiltinFunction::Primitive::Minimum:
		case BuiltinFunction::Primitive::Maximum:
		{
			return ShapeCollector::ShapeFromOperand(inShapes, arguments.at(0));
		}

		// ---------------
		// List Operations
		// ---------------

		//TODO: Use the nested function for geometry computation

		// List
		case BuiltinFunction::Primitive::Each:
		case BuiltinFunction::Primitive::EachItem:
		{
			return ShapeCollector::ShapeFromOperand(inShapes, arguments.at(1));
		}
		case BuiltinFunction::Primitive::EachLeft:
		{
			return ShapeCollector::ShapeFromOperand(inShapes, arguments.at(1));
		}
		case BuiltinFunction::Primitive::EachRight:
		{
			return ShapeCollector::ShapeFromOperand(inShapes, arguments.at(2));
		}

		// ---------------
		// Cell Operations
		// ---------------

		// List
		case BuiltinFunction::Primitive::Raze:
		{
			return ShapeCollector::ShapeFromOperand(inShapes, arguments.at(0));
		}
		case BuiltinFunction::Primitive::ToList:
		{
			const auto& shapes = m_shapeAnalysis.GetShapes(m_call);
			Require(shapes.size() == 1);
			return shapes.at(0);
		}
		
		// --------------
		// CPU Operations
		// --------------

		// Algebraic Unary
		case BuiltinFunction::Primitive::Random:
		case BuiltinFunction::Primitive::Seed:
		case BuiltinFunction::Primitive::Flip:

		// Algebraic Binary
		case BuiltinFunction::Primitive::Replicate:
		case BuiltinFunction::Primitive::Like:

		// List
		case BuiltinFunction::Primitive::List:
		case BuiltinFunction::Primitive::Match:

		// Database
		case BuiltinFunction::Primitive::Enum:
		case BuiltinFunction::Primitive::Dictionary:
		case BuiltinFunction::Primitive::Table:
		case BuiltinFunction::Primitive::KeyedTable:
		case BuiltinFunction::Primitive::Keys:
		case BuiltinFunction::Primitive::Values:
		case BuiltinFunction::Primitive::Meta:
		case BuiltinFunction::Primitive::Fetch:
		case BuiltinFunction::Primitive::ColumnValue:
		case BuiltinFunction::Primitive::LoadTable:

		// Other
		case BuiltinFunction::Primitive::LoadCSV:
		case BuiltinFunction::Primitive::Print:
		case BuiltinFunction::Primitive::String:
		case BuiltinFunction::Primitive::SubString:
		{
			CPU();
		}

		// GPU
		case BuiltinFunction::Primitive::GPUOrderLib:
		case BuiltinFunction::Primitive::GPUGroupLib:
		case BuiltinFunction::Primitive::GPUUniqueLib:
		case BuiltinFunction::Primitive::GPULoopJoinLib:
		case BuiltinFunction::Primitive::GPUHashJoinLib:
		{
			CPU();
		}
		case BuiltinFunction::Primitive::GPUOrderInit:
		{
			const auto& shapes = m_shapeAnalysis.GetShapes(m_call);
			auto indexShape = shapes.at(0);
			Require(ShapeUtils::IsShape<VectorShape>(indexShape));
			return indexShape;
		}
		case BuiltinFunction::Primitive::GPUOrder:
		case BuiltinFunction::Primitive::GPUOrderShared:
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
		case BuiltinFunction::Primitive::GPUGroup:
		case BuiltinFunction::Primitive::GPUUnique:
		{
			auto indexShape = ShapeCollector::ShapeFromOperand(inShapes, arguments.at(0));
			Require(ShapeUtils::IsShape<VectorShape>(indexShape));
			return indexShape;
		}
		case BuiltinFunction::Primitive::GPULoopJoinCount:
		{
			const auto& shapes = m_shapeAnalysis.GetShapes(m_call);
			auto offsetsShape = shapes.at(0);
			Require(ShapeUtils::IsShape<VectorShape>(offsetsShape));
			return offsetsShape;
		}
		case BuiltinFunction::Primitive::GPULoopJoin:
		case BuiltinFunction::Primitive::GPUHashJoinCount:
		case BuiltinFunction::Primitive::GPUHashJoin:
		{
			// Left data geometry

			auto dataIndex = arguments.size();
			switch (function->GetPrimitive())
			{
				// case BuiltinFunction::Primitive::GPULoopJoinCount:
				case BuiltinFunction::Primitive::GPUHashJoinCount:
				{
					dataIndex -= 1;
					break;
				}
				case BuiltinFunction::Primitive::GPULoopJoin:
				case BuiltinFunction::Primitive::GPUHashJoin:
				{
					dataIndex -= 3;
					break;
				}
			}
			auto dataShape = ShapeCollector::ShapeFromOperand(inShapes, arguments.at(dataIndex));
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
		case BuiltinFunction::Primitive::GPUHashCreate:
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

bool GeometryAnalysis::VisitIn(const Operand *operand)
{
	const auto& inShapes = m_shapeAnalysis.GetInSet(m_currentStatement);
	m_currentGeometry = ShapeCollector::ShapeFromOperand(inShapes, operand);
	return false;
}

std::string GeometryAnalysis::DebugString(const Statement *statement, unsigned int indent) const
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
}
