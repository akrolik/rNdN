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
	auto& functionName = function->GetName();

	auto timeGeometry_start = Utils::Chrono::Start(Name + " '" + functionName + "'");
	function->Accept(*this);
	Utils::Chrono::End(timeGeometry_start);

	if (Utils::Options::IsFrontend_PrintAnalysis(ShortName, functionName))
	{
		Utils::Logger::LogInfo(Name + " '" + functionName + "'");

		auto string = StatementAnalysisPrinter::PrettyString(*this, function);
		Utils::Logger::LogInfo(string, 0, true, Utils::Logger::NoPrefix);
	}
}

bool GeometryAnalysis::VisitIn(const DeclarationStatement *declarationS)
{
	m_geometries[declarationS] = m_shapeAnalysis.GetDeclarationShape(declarationS->GetDeclaration());
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
	// Collect input shapes

	const auto& inShapes = m_shapeAnalysis.GetInSet(m_currentStatement);
	const auto& outShapes = m_shapeAnalysis.GetShapes(call);

	std::vector<const Shape *> argumentShapes;
	for (const auto& argument : call->GetArguments())
	{
		argumentShapes.push_back(ShapeCollector::ShapeFromOperand(inShapes, argument));
	}

	// Analyze the call geometry, ignore parameters

	m_currentGeometry = AnalyzeCall(call->GetFunctionLiteral()->GetFunction(), outShapes, argumentShapes, call->GetArguments());

	return false;
}

const Shape *GeometryAnalysis::AnalyzeCall(const FunctionDeclaration *function, const std::vector<const Shape *>& returnShapes, const std::vector<const Shape *>& argumentShapes, const std::vector<const Operand *>& arguments)
{
	switch (function->GetKind())
	{
		case FunctionDeclaration::Kind::Builtin:
			return AnalyzeCall(static_cast<const BuiltinFunction *>(function), returnShapes, argumentShapes, arguments);
		case FunctionDeclaration::Kind::Definition:
			return AnalyzeCall(static_cast<const Function *>(function), returnShapes, argumentShapes, arguments);
		default:
			Utils::Logger::LogError("Unsupported function kind");
	}
}

const Shape *GeometryAnalysis::AnalyzeCall(const Function *function, const std::vector<const Shape *>& returnShapes, const std::vector<const Shape *>& argumentShapes, const std::vector<const Operand *>& arguments)
{
	// Without interprocedural analysis, assume the function operates on an unknown geometry

	return new WildcardShape();
}

const Shape *GeometryAnalysis::AnalyzeCall(const BuiltinFunction *function, const std::vector<const Shape *>& returnShapes, const std::vector<const Shape *>& argumentShapes, const std::vector<const Operand *>& arguments)
{
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
			Require(returnShapes.size() == 1);
			return returnShapes.at(0);
		}

		// --------------------
		// Vector Input Geometry
		// --------------------

		// Indexing
		case BuiltinFunction::Primitive::IndexAssignment:
		{
			// We operate on the size of the index set

			return argumentShapes.at(1);
		}

		// Algebraic Unary
		case BuiltinFunction::Primitive::Unique:
		{
			return argumentShapes.at(0);
		}
		case BuiltinFunction::Primitive::Take:
		case BuiltinFunction::Primitive::Drop:
		{
			// We operate on the size of the input vector

			return argumentShapes.at(1);
		}

		// Algebraic Binary
		case BuiltinFunction::Primitive::Append:
		{
			Require(returnShapes.size() == 1);

			// Only support vector appending on the GPU

			if (const auto vectorShape = ShapeUtils::GetShape<VectorShape>(returnShapes.at(0)))
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
			Require(returnShapes.size() == 1);
			Require(ShapeUtils::IsShape<VectorShape>(returnShapes.at(0)));

			// Recover the full size from the compression

			const auto vectorShape = ShapeUtils::GetShape<VectorShape>(returnShapes.at(0));
			Require(ShapeUtils::IsSize<Shape::CompressedSize>(vectorShape->GetSize()));

			const auto fullSize = ShapeUtils::GetSize<Shape::CompressedSize>(vectorShape->GetSize())->GetSize();
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
			return argumentShapes.at(0);
		}

		// ---------------
		// List Operations
		// ---------------

		// List
		case BuiltinFunction::Primitive::Each:
		{
			const auto type = arguments.at(0)->GetType();
			const auto function = TypeUtils::GetType<FunctionType>(type)->GetFunctionDeclaration();

			const auto argumentShape = argumentShapes.at(1);
			Require(ShapeUtils::IsShape<ListShape>(argumentShape));

			const auto listShape = ShapeUtils::GetShape<ListShape>(argumentShape);

			Require(returnShapes.size() == 1);
			const auto returnShape = returnShapes.at(0);

			Require(ShapeUtils::IsShape<ListShape>(returnShape));
			const auto returnListShape = ShapeUtils::GetShape<ListShape>(returnShape);
			const auto returnElementShapes = returnListShape->GetElementShapes();

			std::vector<const Shape *> elementGeometries;
			auto index = 0u;
			for (const auto& elementShape : listShape->GetElementShapes())
			{
				const auto returnElementShape = returnElementShapes.at(index);
				const auto geometry = AnalyzeCall(function, {returnElementShape}, {elementShape}, {});
				elementGeometries.push_back(geometry);
				index++;
			}
			return new ListShape(listShape->GetListSize(), elementGeometries);
		}
		case BuiltinFunction::Primitive::EachItem:
		{
			const auto type = arguments.at(0)->GetType();
			const auto function = TypeUtils::GetType<FunctionType>(type)->GetFunctionDeclaration();

			const auto argumentShape1 = argumentShapes.at(1);
			const auto argumentShape2 = argumentShapes.at(2);
			Require(ShapeUtils::IsShape<ListShape>(argumentShape1));
			Require(ShapeUtils::IsShape<ListShape>(argumentShape2));

			const auto listShape1 = ShapeUtils::GetShape<ListShape>(argumentShape1);
			const auto listShape2 = ShapeUtils::GetShape<ListShape>(argumentShape2);

			const auto& elementShapes1 = listShape1->GetElementShapes();
			const auto& elementShapes2 = listShape2->GetElementShapes();
			Require(elementShapes1.size() == elementShapes2.size());

			Require(returnShapes.size() == 1);
			const auto returnShape = returnShapes.at(0);

			Require(ShapeUtils::IsShape<ListShape>(returnShape));
			const auto returnListShape = ShapeUtils::GetShape<ListShape>(returnShape);
			const auto returnElementShapes = returnListShape->GetElementShapes();

			std::vector<const Shape *> elementGeometries;
			for (auto i = 0u; i < elementShapes1.size(); ++i)
			{
				const auto returnElementShape = returnElementShapes.at(i);
				const auto elementShape1 = elementShapes1.at(i);
				const auto elementShape2 = elementShapes2.at(i);

				const auto geometry = AnalyzeCall(function, {returnElementShape}, {elementShape1, elementShape2}, {});
				elementGeometries.push_back(geometry);
			}
			return new ListShape(listShape1->GetListSize(), elementGeometries);
		}
		case BuiltinFunction::Primitive::EachLeft:
		{
			const auto type = arguments.at(0)->GetType();
			const auto function = TypeUtils::GetType<FunctionType>(type)->GetFunctionDeclaration();

			const auto argumentShape1 = argumentShapes.at(1);
			const auto argumentShape2 = argumentShapes.at(2);

			Require(ShapeUtils::IsShape<ListShape>(argumentShape1));
			const auto listShape1 = ShapeUtils::GetShape<ListShape>(argumentShape1);

			Require(returnShapes.size() == 1);
			const auto returnShape = returnShapes.at(0);

			Require(ShapeUtils::IsShape<ListShape>(returnShape));
			const auto returnListShape = ShapeUtils::GetShape<ListShape>(returnShape);
			const auto returnElementShapes = returnListShape->GetElementShapes();

			std::vector<const Shape *> elementGeometries;
			auto index = 0u;
			for (const auto& elementShape1 : listShape1->GetElementShapes())
			{
				const auto returnElementShape = returnElementShapes.at(index);
				const auto geometry = AnalyzeCall(function, {returnElementShape}, {elementShape1, argumentShape2}, {});
				elementGeometries.push_back(geometry);
				index++;
			}
			return new ListShape(listShape1->GetListSize(), elementGeometries);
		}
		case BuiltinFunction::Primitive::EachRight:
		{
			const auto type = arguments.at(0)->GetType();
			const auto function = TypeUtils::GetType<FunctionType>(type)->GetFunctionDeclaration();

			const auto argumentShape1 = argumentShapes.at(1);
			const auto argumentShape2 = argumentShapes.at(2);

			Require(ShapeUtils::IsShape<ListShape>(argumentShape2));
			const auto listShape2 = ShapeUtils::GetShape<ListShape>(argumentShape2);

			Require(returnShapes.size() == 1);
			const auto returnShape = returnShapes.at(0);

			Require(ShapeUtils::IsShape<ListShape>(returnShape));
			const auto returnListShape = ShapeUtils::GetShape<ListShape>(returnShape);
			const auto returnElementShapes = returnListShape->GetElementShapes();

			std::vector<const Shape *> elementGeometries;
			auto index = 0u;
			for (const auto& elementShape2 : listShape2->GetElementShapes())
			{
				const auto returnElementShape = returnElementShapes.at(index);
				const auto geometry = AnalyzeCall(function, {returnElementShape}, {argumentShape1, elementShape2}, {});
				elementGeometries.push_back(geometry);
				index++;
			}
			return new ListShape(listShape2->GetListSize(), elementGeometries);
		}

		// ---------------
		// Cell Operations
		// ---------------

		// List
		case BuiltinFunction::Primitive::Raze:
		{
			return argumentShapes.at(0);
		}
		case BuiltinFunction::Primitive::ToList:
		{
			Require(returnShapes.size() == 1);
			return returnShapes.at(0);
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
		case BuiltinFunction::Primitive::GPUHashMemberLib:
		case BuiltinFunction::Primitive::GPULikeLib:
		case BuiltinFunction::Primitive::GPULikeCacheLib:
		{
			CPU();
		}
		case BuiltinFunction::Primitive::GPUOrderInit:
		{
			const auto indexShape = returnShapes.at(0);
			Require(ShapeUtils::IsShape<VectorShape>(indexShape));
			return indexShape;
		}
		case BuiltinFunction::Primitive::GPUOrder:
		case BuiltinFunction::Primitive::GPUOrderShared:
		{
			const auto indexShape = argumentShapes.at(0);
			Require(ShapeUtils::IsShape<VectorShape>(indexShape));

			// Divide by 2 if constant, otherwise return a new dynamic size

			const auto vectorShape = ShapeUtils::GetShape<VectorShape>(indexShape);
			if (const auto constantSize = ShapeUtils::GetSize<Shape::ConstantSize>(vectorShape->GetSize()))
			{
				return new VectorShape(new Shape::ConstantSize(constantSize->GetValue() / 2));
			}
			return new VectorShape(new Shape::DynamicSize(m_call));
		}
		case BuiltinFunction::Primitive::GPUGroup:
		case BuiltinFunction::Primitive::GPUUnique:
		{
			const auto indexShape = argumentShapes.at(0);
			Require(ShapeUtils::IsShape<VectorShape>(indexShape));
			return indexShape;
		}
		case BuiltinFunction::Primitive::GPULoopJoinCount:
		case BuiltinFunction::Primitive::GPUHashJoinCount:
		case BuiltinFunction::Primitive::GPUHashMember:
		{
			const auto offsetsShape = returnShapes.at(0);
			Require(ShapeUtils::IsShape<VectorShape>(offsetsShape));
			return offsetsShape;
		}
		case BuiltinFunction::Primitive::GPULoopJoin:
		case BuiltinFunction::Primitive::GPUHashJoin:
		{
			// Left data geometry

			const auto dataIndex = arguments.size() - 3;
			const auto dataShape = argumentShapes.at(dataIndex);

			if (const auto vectorShape = ShapeUtils::GetShape<VectorShape>(dataShape))
			{
				return vectorShape;
			}
			else if (const auto listShape = ShapeUtils::GetShape<ListShape>(dataShape))
			{
				const auto cellShape = ShapeUtils::MergeShapes(listShape->GetElementShapes());
				Require(ShapeUtils::IsShape<VectorShape>(cellShape));
				return cellShape;
			}
			break;
		}
		case BuiltinFunction::Primitive::GPUHashJoinCreate:
		case BuiltinFunction::Primitive::GPUHashMemberCreate:
		{
			const auto dataShape = argumentShapes.back();
			if (const auto vectorShape = ShapeUtils::GetShape<VectorShape>(dataShape))
			{
				return vectorShape;
			}
			else if (const auto listShape = ShapeUtils::GetShape<ListShape>(dataShape))
			{
				const auto cellShape = ShapeUtils::MergeShapes(listShape->GetElementShapes());
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
	string << std::string(indent * Utils::Logger::IndentSize, ' ');
	string << *m_geometries.at(statement);
	return string.str();
}

}
}
