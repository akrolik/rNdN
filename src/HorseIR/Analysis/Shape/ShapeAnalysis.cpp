#include "HorseIR/Analysis/Shape/ShapeAnalysis.h"

#include "HorseIR/Analysis/Helpers/ValueAnalysisHelper.h"
#include "HorseIR/Analysis/Shape/ShapeUtils.h"

#include "Runtime/DataBuffers/DataBuffer.h"
#include "Runtime/DataBuffers/BufferUtils.h"

#include "Utils/Chrono.h"
#include "Utils/Logger.h"
#include "Utils/Math.h"
#include "Utils/Options.h"

namespace HorseIR {
namespace Analysis {

void ShapeAnalysis::Visit(const Parameter *parameter)
{
	// Add dynamic sized shapes for all parameters

	auto symbol = parameter->GetSymbol();
	if (m_currentSet.first.find(symbol) == m_currentSet.first.end())
	{
		auto shape = ShapeUtils::SymbolicShapeFromType(parameter->GetType(), "param." + parameter->GetName());
		m_currentSet.first[symbol] = shape;
		m_currentSet.second[symbol] = shape;
	}
	if (m_currentSet.second.find(symbol) == m_currentSet.second.end())
	{
		m_currentSet.second[symbol] = m_currentSet.first[symbol];
	}

	m_parameterShapes[parameter] = m_currentSet.first.at(symbol);
}

void ShapeAnalysis::Visit(const DeclarationStatement *declarationS)
{
	// Set initial geometry for variable

	auto declaration = declarationS->GetDeclaration();
	auto shape = ShapeUtils::InitialShapeFromType(declaration->GetType());

	m_currentSet.first[declaration->GetSymbol()] = shape;
	m_currentSet.second[declaration->GetSymbol()] = shape;

	m_declarationShapes[declaration] = shape;
}

void ShapeAnalysis::Visit(const AssignStatement *assignS)
{
	// For each target, update the shape with the shape from the expression

	auto expression = assignS->GetExpression();
	expression->Accept(*this);

	const auto& expressionShapes = GetShapes(expression);
	const auto& writeShapes = GetWriteShapes(expression);

	// Check the number of shapes matches the number of targets

	auto targets = assignS->GetTargets();
	if (expressionShapes.size() != targets.size())
	{
		Utils::Logger::LogError("Mismatched number of shapes for assignment. Received " + std::to_string(expressionShapes.size()) + ", expected " + std::to_string(targets.size()) + ".");
	}
	if (writeShapes.size() != targets.size())
	{
		Utils::Logger::LogError("Mismatched number of write shapes for assignment. Received " + std::to_string(writeShapes.size()) + ", expected " + std::to_string(targets.size()) + ".");
	}

	// Update map for each target symbol

	unsigned int i = 0;
	for (const auto target : targets)
	{
		// Extract the target shape from the expression

		auto symbol = target->GetSymbol();
		m_currentSet.first[symbol] = expressionShapes.at(i);
		m_currentSet.second[symbol] = writeShapes.at(i);
		i++;
	}
}

void ShapeAnalysis::Visit(const BlockStatement *blockS)
{
	// Visit all statements within the block and compute the sets

	ForwardAnalysis<ShapeAnalysisProperties>::Visit(blockS);

	// Kill all declarations that were part of the block

	const auto symbolTable = blockS->GetSymbolTable();
	KillShapes(symbolTable, m_currentSet.first);
	KillShapes(symbolTable, m_currentSet.second);
}

void ShapeAnalysis::KillShapes(const SymbolTable *symbolTable, ::Analysis::Map<SymbolObject, ShapeAnalysisValue>& outMap) const
{
	auto it = outMap.begin();
	while (it != outMap.end())
	{
		auto symbol = it->first;
		if (symbolTable->ContainsSymbol(symbol))
		{
			it = outMap.erase(it);
		}
		else
		{
			++it;
		}
	}
}

void ShapeAnalysis::Visit(const IfStatement *ifS)
{
	ForwardAnalysis<ShapeAnalysisProperties>::Visit(ifS);

	CheckCondition(ifS->GetCondition());
}

void ShapeAnalysis::Visit(const WhileStatement *whileS)
{
	ForwardAnalysis<ShapeAnalysisProperties>::Visit(whileS);

	CheckCondition(whileS->GetCondition());
}

void ShapeAnalysis::Visit(const RepeatStatement *repeatS)
{
	ForwardAnalysis<ShapeAnalysisProperties>::Visit(repeatS);

	CheckCondition(repeatS->GetCondition());
}

void ShapeAnalysis::CheckCondition(const Operand *operand) const
{
	auto conditionShape = GetShape(operand);
	if (!ShapeUtils::IsShape<VectorShape>(conditionShape))
	{
		Utils::Logger::LogError("Condition expects a scalar expression");
	}

	auto conditionSize = ShapeUtils::GetShape<VectorShape>(conditionShape)->GetSize();
	if (ShapeUtils::IsSize<Shape::ConstantSize>(conditionSize))
	{
		if (!ShapeUtils::IsScalarSize(conditionSize))
		{
			Utils::Logger::LogError("Condition expects a scalar expression");
		}
	}
}

void ShapeAnalysis::Visit(const ReturnStatement *returnS)
{
	ForwardAnalysis<ShapeAnalysisProperties>::Visit(returnS);

	std::vector<const Shape *> returnShapes;
	std::vector<const Shape *> returnWriteShapes;
	for (const auto& operand : returnS->GetOperands())
	{
		returnShapes.push_back(GetShape(operand));
		returnWriteShapes.push_back(GetWriteShape(operand));
	}

	if (m_returnShapes.size() == 0)
	{
		m_returnShapes = returnShapes;
		m_returnWriteShapes = returnWriteShapes;
	}
	else
	{
		for (auto i = 0u; i < m_returnShapes.size(); ++i)
		{
			m_returnShapes.at(i) = ShapeUtils::MergeShape(m_returnShapes.at(i), returnShapes.at(i));
			m_returnWriteShapes.at(i) = ShapeUtils::MergeShape(m_returnWriteShapes.at(i), returnWriteShapes.at(i));
		}
	}
}
void ShapeAnalysis::Visit(const CallExpression *call)
{
	m_call = call;

	// Collect shape information for the arguments

	const auto& arguments = call->GetArguments();
	std::vector<const Shape *> argumentShapes;
	for (const auto& argument : arguments)
	{
		argument->Accept(*this);
		argumentShapes.push_back(GetShape(argument));
	}

	// Analyze the function according to the shape rules. Store the call for dynamic sizes

	auto [shapes, writeShapes] = AnalyzeCall(call->GetFunctionLiteral()->GetFunction(), argumentShapes, arguments);

	SetShapes(call, shapes);
	SetWriteShapes(call, writeShapes);

	m_call = nullptr;
}

void ShapeAnalysis::AddCompressionConstraint(const DataObject *dataObject, const Shape::Size *size)
{
	m_compressionConstraints.insert({dataObject, size});
}

bool ShapeAnalysis::CheckStaticScalar(const Shape::Size *size) const
{
	// Check if the size is a static scalar, enforcing if required

	if (ShapeUtils::IsSize<Shape::ConstantSize>(size))
	{
		return ShapeUtils::IsScalarSize(size);
	}
	return !m_enforce;
}

bool ShapeAnalysis::CheckStaticEquality(const Shape::Size *size1, const Shape::Size *size2) const
{
	if (CheckConstrainedEquality(size1, size2))
	{
		return true;
	}
	return !m_enforce;
}

bool ShapeAnalysis::CheckConstrainedEquality(const Shape::Size *size1, const Shape::Size *size2) const
{
	if (*size1 == *size2)
	{
		return true;
	}

	// Check for size equality against the constraints

	if (const auto compressedSize1 = ShapeUtils::GetSize<Shape::CompressedSize>(size1))
	{
		for (const auto& [constraintPredicate, constraintSize] : m_compressionConstraints)
		{
			if (*constraintPredicate == *compressedSize1->GetPredicate())
			{
				return (*constraintSize == *size2);
			}	
		}
	}
	else if (const auto compressedSize2 = ShapeUtils::GetSize<Shape::CompressedSize>(size2))
	{
		for (const auto& [constraintPredicate, constraintSize] : m_compressionConstraints)
		{
			if (*constraintPredicate == *compressedSize2->GetPredicate())
			{
				return (*constraintSize == *size1);
			}	
		}
	}

	return false;
}

bool ShapeAnalysis::CheckStaticTabular(const ListShape *listShape) const
{
	const auto& elementShapes = listShape->GetElementShapes();
	for (const auto& elementShape : elementShapes)
	{
		if (!ShapeUtils::IsShape<VectorShape>(elementShape))
		{
			return false;
		}
	}

	for (const auto& elementShape1 : elementShapes)
	{
		for (const auto& elementShape2 : elementShapes)
		{
			const auto vectorShape1 = ShapeUtils::GetShape<VectorShape>(elementShape1);
			const auto vectorShape2 = ShapeUtils::GetShape<VectorShape>(elementShape2);

			if (!CheckStaticEquality(vectorShape1->GetSize(), vectorShape2->GetSize()))
			{
				return false;
			}
		}
	}

	return true;
}

template<class T>
std::pair<bool, T> ShapeAnalysis::GetConstantArgument(const std::vector<const Operand *>& arguments, unsigned int index) const
{
	// Return {found, value}

	if (index < arguments.size())
	{
		// Get constant literal arguments

		if (ValueAnalysisHelper::IsConstant(arguments.at(index)))
		{
			return {true, ValueAnalysisHelper::GetScalar<T>(arguments.at(index))};
		}

		// Get dynamic buffer arguments

		if (const auto buffer = m_dataAnalysis.GetDataObject(arguments.at(index))->GetDataBuffer())
		{
			Utils::ScopedChrono bufferChrono("Runtime analysis data");

			const auto vectorBuffer = Runtime::BufferUtils::GetBuffer<Runtime::VectorBuffer>(buffer);
			if (vectorBuffer->GetElementCount() == 1)
			{
				switch (vectorBuffer->GetType()->GetBasicKind())
				{
					case BasicType::BasicKind::Boolean:
					case BasicType::BasicKind::Char:
					case BasicType::BasicKind::Int8:
					{
						return {true, Runtime::BufferUtils::GetVectorBuffer<std::int8_t>(buffer)->GetCPUReadBuffer()->GetValue<T>(0)};
					}
					case BasicType::BasicKind::Int16:
					{
						return {true, Runtime::BufferUtils::GetVectorBuffer<std::int16_t>(buffer)->GetCPUReadBuffer()->GetValue<T>(0)};
					}
					case BasicType::BasicKind::Int32:
					{
						return {true, Runtime::BufferUtils::GetVectorBuffer<std::int32_t>(buffer)->GetCPUReadBuffer()->GetValue<T>(0)};
					}
					case BasicType::BasicKind::Int64:
					{
						return {true, Runtime::BufferUtils::GetVectorBuffer<std::int64_t>(buffer)->GetCPUReadBuffer()->GetValue<T>(0)};
					}
					case BasicType::BasicKind::Float32:
					{
						return {true, Runtime::BufferUtils::GetVectorBuffer<float>(buffer)->GetCPUReadBuffer()->GetValue<T>(0)};
					}
					case BasicType::BasicKind::Float64:
					{
						return {true, Runtime::BufferUtils::GetVectorBuffer<double>(buffer)->GetCPUReadBuffer()->GetValue<T>(0)};
					}
					case BasicType::BasicKind::Complex:
					{
						return {true, Runtime::BufferUtils::GetVectorBuffer<const ComplexValue *>(buffer)->GetCPUReadBuffer()->GetValue<T>(0)};
					}
					case BasicType::BasicKind::Symbol:
					{
						return {true, Runtime::BufferUtils::GetVectorBuffer<const SymbolValue *>(buffer)->GetCPUReadBuffer()->GetValue<T>(0)};
					}
					case BasicType::BasicKind::String:
					{
						return {true, Runtime::BufferUtils::GetVectorBuffer<std::string>(buffer)->GetCPUReadBuffer()->GetValue<T>(0)};
					}
					case BasicType::BasicKind::Datetime:
					{
						return {true, Runtime::BufferUtils::GetVectorBuffer<const DatetimeValue *>(buffer)->GetCPUReadBuffer()->GetValue<T>(0)};
					}
					case BasicType::BasicKind::Date:
					{
						return {true, Runtime::BufferUtils::GetVectorBuffer<const DateValue *>(buffer)->GetCPUReadBuffer()->GetValue<T>(0)};
					}
					case BasicType::BasicKind::Month:
					{
						return {true, Runtime::BufferUtils::GetVectorBuffer<const MonthValue *>(buffer)->GetCPUReadBuffer()->GetValue<T>(0)};
					}
					case BasicType::BasicKind::Minute:
					{
						return {true, Runtime::BufferUtils::GetVectorBuffer<const MinuteValue *>(buffer)->GetCPUReadBuffer()->GetValue<T>(0)};
					}
					case BasicType::BasicKind::Second:
					{
						return {true, Runtime::BufferUtils::GetVectorBuffer<const SecondValue *>(buffer)->GetCPUReadBuffer()->GetValue<T>(0)};
					}
					case BasicType::BasicKind::Time:
					{
						return {true, Runtime::BufferUtils::GetVectorBuffer<const TimeValue *>(buffer)->GetCPUReadBuffer()->GetValue<T>(0)};
					}
				}
			}
		}
	}

	// Error if the shape is enforced

	if (m_enforce)
	{
		Utils::Logger::LogError("Shape analysis expected constant argument");
	}
	return {false, 0};
}

std::pair<std::vector<const Shape *>, std::vector<const Shape *>> ShapeAnalysis::AnalyzeCall(const FunctionDeclaration *function, const std::vector<const Shape *>& argumentShapes, const std::vector<const Operand *>& arguments)
{
	switch (function->GetKind())
	{
		case FunctionDeclaration::Kind::Builtin:
			return AnalyzeCall(static_cast<const BuiltinFunction *>(function), argumentShapes, arguments);
		case FunctionDeclaration::Kind::Definition:
			return AnalyzeCall(static_cast<const Function *>(function), argumentShapes, arguments);
		default:
			Utils::Logger::LogError("Unsupported function kind");
	}
}

std::pair<std::vector<const Shape *>, std::vector<const Shape *>> ShapeAnalysis::AnalyzeCall(const Function *function, const std::vector<const Shape *>& argumentShapes, const std::vector<const Operand *>& arguments)
{
	// Collect the input shapes for the function

	Properties inputShapes;
	for (auto i = 0u; i < argumentShapes.size(); ++i)
	{
		const auto symbol = function->GetParameter(i)->GetSymbol();
		const auto shape = argumentShapes.at(i);
		inputShapes.first[symbol] = shape;
		inputShapes.second[symbol] = shape;
	}

	// Interprocedural analysis

	const auto& dataAnalysis = m_dataAnalysis.GetAnalysis(function);
	auto shapeAnalysis = new ShapeAnalysis(dataAnalysis, m_program, m_enforce);

	Utils::Chrono::Pause(m_functionTime);
	shapeAnalysis->Analyze(function, inputShapes);
	Utils::Chrono::Continue(m_functionTime);

	m_interproceduralMap.insert({function, shapeAnalysis});

	auto returnShapes = shapeAnalysis->GetReturnShapes();
	return {returnShapes, returnShapes};
}

std::pair<std::vector<const Shape *>, std::vector<const Shape *>> ShapeAnalysis::AnalyzeCall(const BuiltinFunction *function, const std::vector<const Shape *>& argumentShapes, const std::vector<const Operand *>& arguments)
{
	switch (function->GetPrimitive())
	{
#define Require(x) if (!(x)) break
#define Return(x) { std::vector<const Shape *> shapes({x}); return {shapes, shapes}; }
#define Return2(x,y) { std::vector<const Shape *> shapes({x,y}); return {shapes, shapes}; }

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
		{
			// -- Propagate size
			// Input: Vector<Size1*>
			// Output: Vector<Size1*>

			const auto argumentShape = argumentShapes.at(0);
			Require(ShapeUtils::IsShape<VectorShape>(argumentShape));
			Return(argumentShape);
		}

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
		{
			// -- Equality, propagate
			// Input: Vector<Size*>, Vector<Size*>
			// Output: Vector<Size*>
			//
			// -- Broadcast
			// Input: Vector<1>, Vector<Size*> // Vector<Size*>, Vector<1>
			// Output: Vector<Size*>
			//
			// -- Unknown
			// Input: Vector<Size1*>, Vector<Size2*>
			// Output: Vector<SizeDynamic>

			const auto argumentShape1 = argumentShapes.at(0);
			const auto argumentShape2 = argumentShapes.at(1);
			Require(ShapeUtils::IsShape<VectorShape>(argumentShape1));
			Require(ShapeUtils::IsShape<VectorShape>(argumentShape2));

			const auto argumentSize1 = ShapeUtils::GetShape<VectorShape>(argumentShape1)->GetSize();
			const auto argumentSize2 = ShapeUtils::GetShape<VectorShape>(argumentShape2)->GetSize();

			const Shape::Size *size = nullptr;

			if (CheckConstrainedEquality(argumentSize1, argumentSize2))
			{
				size = argumentSize1;
			}
			else if (ShapeUtils::IsScalarSize(argumentSize1))
			{
				// Broadcast to right

				size = argumentSize2;
			}
			else if (ShapeUtils::IsScalarSize(argumentSize2))
			{
				// Broadcast to left

				size = argumentSize1;
			}
			else if (ShapeUtils::IsSize<Shape::ConstantSize>(argumentSize1) && ShapeUtils::IsSize<Shape::ConstantSize>(argumentSize2))
			{
				// Error case, unequal constants where neither is a scalar

				auto constant1 = ShapeUtils::GetSize<Shape::ConstantSize>(argumentSize1)->GetValue();
				auto constant2 = ShapeUtils::GetSize<Shape::ConstantSize>(argumentSize2)->GetValue();
				Utils::Logger::LogError("Binary function '" + function->GetName() + "' requires vector of same length (or broadcast) [" + std::to_string(constant1) + " != " + std::to_string(constant2) + "]");
			}
			else
			{
				// Determine at runtime

				Require(!m_enforce);
				size = new Shape::DynamicSize(m_call);
			}
			Return(new VectorShape(size));
		}

		// Algebraic Unary
		case BuiltinFunction::Primitive::Unique:
		{
			// -- Output is always dynamic
			// Input: Vector<Size*>
			// Output: Vector<SizeDynamic>

			const auto argumentShape = argumentShapes.at(0);
			Require(ShapeUtils::IsShape<VectorShape>(argumentShape));

			const auto vectorShape = ShapeUtils::GetShape<VectorShape>(argumentShape);
			Return(new VectorShape(new Shape::CompressedSize(new DataObject(), vectorShape->GetSize())));
		}
		case BuiltinFunction::Primitive::Range:
		{
			// -- Unknown scalar constant
			// Intput: Vector<1>
			// Output: Vector<SizeDynamic>
			//
			// -- Known scalar constant, create static range
			// Intput: Vector<1> (value k)
			// Iutput: Vector<k>

			const auto argumentShape = argumentShapes.at(0);
			Require(ShapeUtils::IsShape<VectorShape>(argumentShape));

			const auto vectorShape = ShapeUtils::GetShape<VectorShape>(argumentShape);
			Require(CheckStaticScalar(vectorShape->GetSize()));

			if (const auto [isConstant, value] = GetConstantArgument<std::int64_t>(arguments, 0); isConstant)
			{
				Return(new VectorShape(new Shape::ConstantSize(value)));
			}
			Return(new VectorShape(new Shape::DynamicSize(m_call)));
		}
		case BuiltinFunction::Primitive::Factorial:
		case BuiltinFunction::Primitive::Reverse:
		{
			// -- Propagate size
			// Input: Vector<Size*>
			// Output: Vector<Size*>

			const auto argumentShape = argumentShapes.at(0);
			Require(ShapeUtils::IsShape<VectorShape>(argumentShape));
			Return(argumentShape);
		}
		case BuiltinFunction::Primitive::Random:
		case BuiltinFunction::Primitive::Seed:
		{
			// -- Scalar
			// Input: Vector<1>
			// Output: Vector<1>

			const auto argumentShape = argumentShapes.at(0);
			Require(ShapeUtils::IsShape<VectorShape>(argumentShape));

			const auto vectorShape = ShapeUtils::GetShape<VectorShape>(argumentShape);
			Require(CheckStaticScalar(vectorShape->GetSize()));

			Return(new VectorShape(new Shape::ConstantSize(1)));
		}
		case BuiltinFunction::Primitive::Flip:
		{
			// - Propagate size and flip
			// Input: List<Size*, {Shape1*, ..., ShapeN*}>
			// Output: List<Size*, {ShapeN*, ..., Shape1*}>

			const auto argumentShape = argumentShapes.at(0);
			Require(ShapeUtils::IsShape<ListShape>(argumentShape));

			const auto listShape = ShapeUtils::GetShape<ListShape>(argumentShape);

			// Copied intentionally to reverse

			auto elementShapes = listShape->GetElementShapes();
			std::reverse(std::begin(elementShapes), std::end(elementShapes));

			Return(new ListShape(listShape->GetListSize(), elementShapes));
		}
		case BuiltinFunction::Primitive::Where:
		{
			// -- Compress itself(!) by the mask
			// Input: Vector<Size*> (mask)
			// Output: Vector<Size*[mask]>

			const auto argumentShape = argumentShapes.at(0);
			Require(ShapeUtils::IsShape<VectorShape>(argumentShape));

			const auto dataObject = m_dataAnalysis.GetDataObject(arguments.at(0));
			const auto argumentSize = ShapeUtils::GetShape<VectorShape>(argumentShape)->GetSize();
			Return(new VectorShape(new Shape::CompressedSize(dataObject, argumentSize)));
		}
		case BuiltinFunction::Primitive::Group:
		{
			// -- Vector/list group
			// Input: Vector<Size*> // List<Size*, {Shape*}>
			// Output: Dictionary<Vector<SizeDynamic1>, Vector<SizeDynamic2>>
			//
			// For lists, ensure all shapes are vectors of the same length

			const auto argumentShape = argumentShapes.at(0);
			Require(ShapeUtils::IsShape<VectorShape>(argumentShape) || ShapeUtils::IsShape<ListShape>(argumentShape));

			if (const auto listShape = ShapeUtils::GetShape<ListShape>(argumentShape))
			{
				Require(CheckStaticTabular(listShape));
			}
			Return(new DictionaryShape(
				new VectorShape(new Shape::DynamicSize(m_call, 1)),
				new VectorShape(new Shape::DynamicSize(m_call, 2))
			));
		}

		// Algebraic Binary
		case BuiltinFunction::Primitive::Append:
		{
			// -- Vector appending
			// Input: Vector<k1>, Vector<k2>
			// Output: Vector<k1 + k2>
			//
			// Input: Vector<Size1*>, Vector<Size2*>
			// Output: Vector<SizeDynamic>
			// 
			// -- List appending
			// Input: List<k1, {Shapes1*}>, List<k2, {Shapes2*}>
			// Output: List<k1 + k2, {Shapes1*}U{Shapes2*}>
			//
			// Input: List<k1, {Shape1*}>, List<k2, {Shape1*}>
			// Output: List<k1 + k2, {Shape1*}>
			//
			// Input: List<Size1*, {Shapes1*}>, List<Size2*, {Shapes2*}>
			// Output: List<SizeDynamic, {Merge(Shapes1*, Shapes2*)}>
			//
			// -- Enumeration appending
			// Input: Enum<Vector<Size*>, Vector<k1>>, Vector<k2>
			// Output: Enum<Vector<Size*>, Vector<k1 + k2>>
			//
			// Input: Enum<Vector<Size1*>, Vector<Size2*>>, Vector<Size3*>
			// Output:Enum<Vector<Size1*>, Vector<SizeDynamic>>
			//
			// Input: Enum<List<Size*, {Shapes1*}>, List<Size*, {Shapes2*}>>, List<Size*, {Shapes3*}>
			// Output: Enum<List<Size*, {Shapes1*}>, List<Size*, {Merge(Shapes2*, Shapes3*)}>>

			const auto argumentShape1 = argumentShapes.at(0);
			const auto argumentShape2 = argumentShapes.at(1);

			if (ShapeUtils::IsShape<VectorShape>(argumentShape1) && ShapeUtils::IsShape<VectorShape>(argumentShape2))
			{
				const auto vectorSize1 = ShapeUtils::GetShape<VectorShape>(argumentShape1)->GetSize();
				const auto vectorSize2 = ShapeUtils::GetShape<VectorShape>(argumentShape2)->GetSize();

				if (ShapeUtils::IsSize<Shape::ConstantSize>(vectorSize1) && ShapeUtils::IsSize<Shape::ConstantSize>(vectorSize2))
				{
					auto length1 = ShapeUtils::GetSize<Shape::ConstantSize>(vectorSize1)->GetValue();
					auto length2 = ShapeUtils::GetSize<Shape::ConstantSize>(vectorSize2)->GetValue();
					Return(new VectorShape(new Shape::ConstantSize(length1 + length2)));
				}
				Return(new VectorShape(new Shape::DynamicSize(m_call)));
			}
			else if (ShapeUtils::IsShape<ListShape>(argumentShape1) || ShapeUtils::IsShape<ListShape>(argumentShape2))
			{
				const auto listShape1 = ShapeUtils::GetShape<ListShape>(argumentShape1);
				const auto listShape2 = ShapeUtils::GetShape<ListShape>(argumentShape2);

				const auto& elementShapes1 = (listShape1 == nullptr) ? std::vector<const Shape *>({argumentShape1}) : listShape1->GetElementShapes();
				const auto& elementShapes2 = (listShape2 == nullptr) ? std::vector<const Shape *>({argumentShape2}) : listShape2->GetElementShapes();

				const auto listSize1 = (listShape1 == nullptr) ? nullptr : listShape1->GetListSize();
				const auto listSize2 = (listShape2 == nullptr) ? nullptr : listShape2->GetListSize();

				bool constant1 = (listSize1 == nullptr || ShapeUtils::IsSize<Shape::ConstantSize>(listSize1));
				bool constant2 = (listSize2 == nullptr || ShapeUtils::IsSize<Shape::ConstantSize>(listSize2));
				if (constant1 && constant2)
				{
					auto length1 = (listSize1 == nullptr) ? 1 : ShapeUtils::GetSize<Shape::ConstantSize>(listSize1)->GetValue();
					auto length2 = (listSize1 == nullptr) ? 1 : ShapeUtils::GetSize<Shape::ConstantSize>(listSize1)->GetValue();

					std::vector<const Shape *> elementShapes;
					if (elementShapes1.size() == length1 && elementShapes2.size() == length2)
					{
						elementShapes.insert(std::begin(elementShapes), std::begin(elementShapes1), std::end(elementShapes1));
						elementShapes.insert(std::begin(elementShapes), std::begin(elementShapes2), std::end(elementShapes2));
					}
					else
					{
						auto mergedShapes1 = ShapeUtils::MergeShapes(elementShapes1);
						auto mergedShapes2 = ShapeUtils::MergeShapes(elementShapes2);
						elementShapes.push_back(ShapeUtils::MergeShape(mergedShapes1, mergedShapes2));
					}
					Return(new ListShape(new Shape::ConstantSize(length1 + length2), elementShapes));
				}

				auto mergedShapes1 = ShapeUtils::MergeShapes(elementShapes1);
				auto mergedShapes2 = ShapeUtils::MergeShapes(elementShapes2);
				auto mergedShape = ShapeUtils::MergeShape(mergedShapes1, mergedShapes2);
				Return(new ListShape(new Shape::DynamicSize(m_call), {mergedShape}));
			}
			else if (const auto enumShape = ShapeUtils::GetShape<EnumerationShape>(argumentShape1))
			{
				const auto keyShape = enumShape->GetKeyShape();
				const auto valueShape = enumShape->GetValueShape();
				if (const auto vectorValueShape = ShapeUtils::GetShape<VectorShape>(valueShape))
				{
					Require(ShapeUtils::IsShape<VectorShape>(argumentShape2));

					const auto vectorSize1 = vectorValueShape->GetSize();
					const auto vectorSize2 = ShapeUtils::GetShape<VectorShape>(argumentShape2)->GetSize();

					if (ShapeUtils::IsSize<Shape::ConstantSize>(vectorSize1) && ShapeUtils::IsSize<Shape::ConstantSize>(vectorSize2))
					{
						auto length1 = ShapeUtils::IsSize<Shape::ConstantSize>(vectorSize1);
						auto length2 = ShapeUtils::IsSize<Shape::ConstantSize>(vectorSize2);
						Return(new EnumerationShape(keyShape, new VectorShape(new Shape::ConstantSize(length1 + length2))));
					}
					Return(new EnumerationShape(keyShape, new VectorShape(new Shape::DynamicSize(m_call))));
				}
				else if (const auto listValueShape = ShapeUtils::GetShape<ListShape>(valueShape))
				{
					Require(ShapeUtils::IsShape<ListShape>(argumentShape2));

					const auto listShape2 = ShapeUtils::GetShape<ListShape>(argumentShape2);
					Require(CheckStaticEquality(listValueShape->GetListSize(), listShape2->GetListSize()));
					Require(CheckStaticTabular(listShape2));

					const auto& elementShapes1 = listValueShape->GetElementShapes();
					const auto& elementShapes2 = listShape2->GetElementShapes();
					if (elementShapes1.size() == elementShapes2.size())
					{
						std::vector<const Shape *> newElementShapes;
						for (auto i = 0u; i < elementShapes1.size(); ++i)
						{
							const auto elementShape1 = elementShapes1.at(0);
							const auto elementShape2 = elementShapes2.at(1);
							Require(ShapeUtils::IsShape<VectorShape>(elementShape1));
							Require(ShapeUtils::IsShape<VectorShape>(elementShape2));

							const auto vectorSize1 = ShapeUtils::GetShape<VectorShape>(elementShape1)->GetSize();
							const auto vectorSize2 = ShapeUtils::GetShape<VectorShape>(elementShape2)->GetSize();
							if (ShapeUtils::IsSize<Shape::ConstantSize>(vectorSize1) && ShapeUtils::IsSize<Shape::ConstantSize>(vectorSize2))
							{
								auto length1 = ShapeUtils::GetSize<Shape::ConstantSize>(vectorSize1)->GetValue();
								auto length2 = ShapeUtils::GetSize<Shape::ConstantSize>(vectorSize2)->GetValue();
								newElementShapes.push_back(new VectorShape(new Shape::ConstantSize(length1 + length2)));
							}
							else
							{
								newElementShapes.push_back(new VectorShape(new Shape::DynamicSize(m_call, i+1)));
							}
						}
						Return(new EnumerationShape(keyShape, new ListShape(listValueShape->GetListSize(), newElementShapes)));
					}
					Return(new EnumerationShape(keyShape, new ListShape(listValueShape->GetListSize(), {new VectorShape(new Shape::DynamicSize(m_call))})));
				}
			}
			break;
		}
		case BuiltinFunction::Primitive::Replicate:
		{
			// -- Vector input, static value
			// Input: Vector<1> (value k), Vector<Size*>
			// Output: List<k, {Vector<Size*>}>
			//
			// -- Vector input, dynamic value
			// Input: Vector<1>, Vector<Size*>
			// Output: List<SizeDynamic, {Vector<Size*>}>
			//
			// -- List input, static value
			// Input: Vector<1> (value k), List<Size*, {Shape*}>
			// Output: List<k x Size*, {Shape*}>
			//
			// -- List input, dynamic value
			// Input: Vector<1>, List<Size*, {Shape*}>
			// Output: List<DynamicSize, {Shape*}>

			const auto argumentShape1 = argumentShapes.at(0);
			const auto argumentShape2 = argumentShapes.at(1);
			Require(ShapeUtils::IsShape<VectorShape>(argumentShape1));
			Require(ShapeUtils::IsShape<VectorShape>(argumentShape2) || ShapeUtils::IsShape<ListShape>(argumentShape2));

			const auto vectorShape1 = ShapeUtils::GetShape<VectorShape>(argumentShape1);
			Require(CheckStaticScalar(vectorShape1->GetSize()));

			if (const auto [isConstant, value] = GetConstantArgument<std::int64_t>(arguments, 0); isConstant)
			{
				if (ShapeUtils::IsShape<VectorShape>(argumentShape2))
				{
					Return(new ListShape(new Shape::ConstantSize(value), {argumentShape2}));
				}
				else if (const auto listShape = ShapeUtils::GetShape<ListShape>(argumentShape2))
				{
					if (const auto constantSize = ShapeUtils::GetSize<Shape::ConstantSize>(listShape->GetListSize()))
					{
						auto listLength = constantSize->GetValue();

						const auto& elementShapes = listShape->GetElementShapes();
						if (elementShapes.size() == 1)
						{
							Return(new ListShape(new Shape::ConstantSize(value * listLength), {elementShapes.at(0)}));
						}

						std::vector<const Shape *> newElementShapes;
						for (auto i = 0u; i < listLength; ++i)
						{
							newElementShapes.insert(std::end(newElementShapes), std::begin(elementShapes), std::end(elementShapes));
						}
						Return(new ListShape(new Shape::ConstantSize(value * listLength), newElementShapes));
					}
					Return(new ListShape(new Shape::DynamicSize(m_call), {ShapeUtils::MergeShapes(listShape->GetElementShapes())}));
				}
			}
			else
			{
				if (ShapeUtils::IsShape<VectorShape>(argumentShape2))
				{
					Return(new ListShape(new Shape::DynamicSize(m_call), {argumentShape2}));
				}
				else if (const auto listShape = ShapeUtils::GetShape<ListShape>(argumentShape2))
				{
					const auto elementShape = ShapeUtils::MergeShapes(listShape->GetElementShapes());
					Return(new ListShape(new Shape::DynamicSize(m_call), {elementShape}));
				}
			}
			break;
		}
		case BuiltinFunction::Primitive::Like:
		{
			// -- Propagate the left vector size
			// Input: Vector<Size*>, Vector<1>
			// Output: Vector<Size*>

			const auto argumentShape1 = argumentShapes.at(0);
			const auto argumentShape2 = argumentShapes.at(1);
			Require(ShapeUtils::IsShape<VectorShape>(argumentShape1));
			Require(ShapeUtils::IsShape<VectorShape>(argumentShape2));

			const auto vectorShape2 = ShapeUtils::GetShape<VectorShape>(argumentShape2);
			Require(CheckStaticScalar(vectorShape2->GetSize()));
			Return(argumentShape1);
		}
		case BuiltinFunction::Primitive::Compress:
		{
			// -- Compress the right shape by the mask
			// Input: Vector<Size*> (mask), Vector<Size*>
			// Output: Vector<Size*[mask]>

			const auto argumentShape1 = argumentShapes.at(0);
			const auto argumentShape2 = argumentShapes.at(1);
			Require(ShapeUtils::IsShape<VectorShape>(argumentShape1));
			Require(ShapeUtils::IsShape<VectorShape>(argumentShape2));

			const auto argumentSize1 = ShapeUtils::GetShape<VectorShape>(argumentShape1)->GetSize();
			const auto argumentSize2 = ShapeUtils::GetShape<VectorShape>(argumentShape2)->GetSize();
			Require(CheckStaticEquality(argumentSize1, argumentSize2));

			const auto dataObject = m_dataAnalysis.GetDataObject(arguments.at(0));
			Return(new VectorShape(new Shape::CompressedSize(dataObject, argumentSize2)));
		}
		case BuiltinFunction::Primitive::Random_k:
		{
			// -- Static scalar constant, create value range
			// Input: Vector<1> (value k), Vector<1>
			// Output: Vector<k>
			//
			// -- Unknown scalar constant
			// Intput: Vector<1>, Vector<1>
			// Output: Vector<SizeDynamic>

			const auto argumentShape1 = argumentShapes.at(0);
			const auto argumentShape2 = argumentShapes.at(1);
			Require(ShapeUtils::IsShape<VectorShape>(argumentShape1));
			Require(ShapeUtils::IsShape<VectorShape>(argumentShape2));

			const auto vectorShape1 = ShapeUtils::GetShape<VectorShape>(argumentShape1);
			const auto vectorShape2 = ShapeUtils::GetShape<VectorShape>(argumentShape2);
			Require(CheckStaticScalar(vectorShape1->GetSize()));
			Require(CheckStaticScalar(vectorShape2->GetSize()));

			if (const auto [isConstant, value] = GetConstantArgument<std::int64_t>(arguments, 0); isConstant)
			{
				Return(new VectorShape(new Shape::ConstantSize(value)));
			}
			Return(new VectorShape(new Shape::DynamicSize(m_call)));
		}
		case BuiltinFunction::Primitive::IndexOf:
		{
			// -- Propagate right
			// Input: Vector<Size1*>, Vector<Size2*>
			// Output: Vector<Size2*>

			const auto argumentShape1 = argumentShapes.at(0);
			const auto argumentShape2 = argumentShapes.at(1);
			Require(ShapeUtils::IsShape<VectorShape>(argumentShape1));
			Require(ShapeUtils::IsShape<VectorShape>(argumentShape2));
			Return(argumentShape1);
		}
		case BuiltinFunction::Primitive::Take:
		{
			// -- Static scalar constant
			// Input: Vector<1> (value k), Vector<Size*>
			// Output: Vector<k>
			//
			// -- Unknown scalar constant
			// Intput: Vector<1>, Vector<Size*>
			// Output: Vector<SizeDynamic>

			const auto argumentShape1 = argumentShapes.at(0);
			const auto argumentShape2 = argumentShapes.at(1);
			Require(ShapeUtils::IsShape<VectorShape>(argumentShape1));
			Require(ShapeUtils::IsShape<VectorShape>(argumentShape2));

			const auto vectorShape1 = ShapeUtils::GetShape<VectorShape>(argumentShape1);
			const auto vectorShape2 = ShapeUtils::GetShape<VectorShape>(argumentShape2);
			Require(CheckStaticScalar(vectorShape1->GetSize()));

			const auto writeShape = new VectorShape(new Shape::CompressedSize(new DataObject(), vectorShape2->GetSize()));
			if (const auto [isConstant, value] = GetConstantArgument<std::int64_t>(arguments, 0); isConstant)
			{
				auto absValue = (value < 0) ? -value : value;
				return {{new VectorShape(new Shape::ConstantSize(absValue))}, {writeShape}};
			}
			return {{new VectorShape(new Shape::DynamicSize(m_call))}, {writeShape}};
		}
		case BuiltinFunction::Primitive::Drop:
		{
			// -- Static scalar constant and vector length
			// Input: Vector<1> (value k), Vector<SizeConstant>
			// Output: Vector<SizeConstant - k> (ceil 0)
			//
			// -- Unknown scalar constant or vector length
			// Intput: Vector<1>, Vector<Size*>
			// Output: Vector<SizeDynamic>

			const auto argumentShape1 = argumentShapes.at(0);
			const auto argumentShape2 = argumentShapes.at(1);
			Require(ShapeUtils::IsShape<VectorShape>(argumentShape1));
			Require(ShapeUtils::IsShape<VectorShape>(argumentShape2));

			const auto vectorShape1 = ShapeUtils::GetShape<VectorShape>(argumentShape1);
			const auto vectorShape2 = ShapeUtils::GetShape<VectorShape>(argumentShape2);
			Require(CheckStaticScalar(vectorShape1->GetSize()));

			const auto writeShape = new VectorShape(new Shape::CompressedSize(new DataObject(), vectorShape2->GetSize()));
			if (const auto [isConstant, modLength] = GetConstantArgument<std::int64_t>(arguments, 0); isConstant)
			{
				if (const auto constantSize = ShapeUtils::GetSize<Shape::ConstantSize>(vectorShape2->GetSize()))
				{
					// Compute the modification length

					auto absModLength = (modLength < 0) ? -modLength : modLength;

					// Compute the new vector length, ceil at 0

					auto vectorLength = constantSize->GetValue();
					auto value = vectorLength - absModLength;
					value = (value < 0) ? 0 : value;

					return {{new VectorShape(new Shape::ConstantSize(value))}, {writeShape}};
				}
			}
			return {{new VectorShape(new Shape::DynamicSize(m_call))}, {writeShape}};
		}
		case BuiltinFunction::Primitive::Order:
		{
			// -- Vector input
			// Input: Vector<Size*>, Vector<1>
			// Output: Vector<Size*>
			//
			// -- List input
			// Input: List<k, {Vector<Size*>}>, Vector<k>
			// Output: Vector<Size*>

			const auto argumentShape1 = argumentShapes.at(0);
			const auto argumentShape2 = argumentShapes.at(1);
			Require(ShapeUtils::IsShape<VectorShape>(argumentShape2));

			const auto vectorShape2 = ShapeUtils::GetShape<VectorShape>(argumentShape2);
			if (ShapeUtils::IsShape<VectorShape>(argumentShape1))
			{
				Require(CheckStaticScalar(vectorShape2->GetSize()));
				Return(argumentShape1);
			}
			else if (const auto listShape = ShapeUtils::GetShape<ListShape>(argumentShape1))
			{
				Require(CheckStaticEquality(listShape->GetListSize(), vectorShape2->GetSize()) || CheckStaticScalar(vectorShape2->GetSize()));
				Require(CheckStaticTabular(listShape));
				Return(ShapeUtils::MergeShapes(listShape->GetElementShapes()));
			}
			break;
		}
		case BuiltinFunction::Primitive::Member:
		{
			// -- Propagate left shape
			// Input: Vector<Size1*>, Vector<Size2*>
			// Output: Vector<Size1*>

			const auto argumentShape1 = argumentShapes.at(0);
			const auto argumentShape2 = argumentShapes.at(1);
			Require(ShapeUtils::IsShape<VectorShape>(argumentShape1));
			Require(ShapeUtils::IsShape<VectorShape>(argumentShape2));
			Return(argumentShape1);
		}
		case BuiltinFunction::Primitive::Vector:
		{
			// -- Static scalar constant for size
			// Input: Vector<1> (value k), Vector<Size*>
			// Output: Vector<k>
			//
			// -- Unknown scalar constant
			// Intput: Vector<1>, Vector<Size*>
			// Output: Vector<SizeDynamic>

			const auto argumentShape1 = argumentShapes.at(0);
			const auto argumentShape2 = argumentShapes.at(1);
			Require(ShapeUtils::IsShape<VectorShape>(argumentShape1));
			Require(ShapeUtils::IsShape<VectorShape>(argumentShape2));

			const auto vectorShape1 = ShapeUtils::GetShape<VectorShape>(argumentShape1);
			Require(CheckStaticScalar(vectorShape1->GetSize()));

			if (const auto [isConstant, value] = GetConstantArgument<std::int64_t>(arguments, 0); isConstant)
			{
				Return(new VectorShape(new Shape::ConstantSize(value)));
			}
			Return(new VectorShape(new Shape::DynamicSize(m_call)));
		}

		// Reduction
		case BuiltinFunction::Primitive::Length:
		case BuiltinFunction::Primitive::Sum:
		case BuiltinFunction::Primitive::Average:
		case BuiltinFunction::Primitive::Minimum:
		case BuiltinFunction::Primitive::Maximum:
		{
			// -- Reduce to a single value
			// Input: Vector<Size*>
			// Output: Vector<1>

			const auto argumentShape = argumentShapes.at(0);
			Require(ShapeUtils::IsShape<VectorShape>(argumentShape));
			Return(new VectorShape(new Shape::ConstantSize(1)));
		}

		// List
		case BuiltinFunction::Primitive::Raze:
		{
			// -- Static reduce list to vector
			// Input: List<k, {Vector<k1>, ..., Vector<kk>}>
			// Output: Vector<k1 + ... + kk>
			//
			// -- Dynamic reduce list to vector
			// Input: List<Size*, {Vector<Size1*>, ..., Vector<Sizek*>}>
			// Output: Vector<SizeDynamic>

			const auto argumentShape = argumentShapes.at(0);
			Require(ShapeUtils::IsShape<ListShape>(argumentShape));

			const auto listShape = ShapeUtils::GetShape<ListShape>(argumentShape);
			if (const auto listSize = ShapeUtils::GetSize<Shape::ConstantSize>(listShape->GetListSize()))
			{
				const auto& elementShapes = listShape->GetElementShapes();
				if (elementShapes.size() == 1)
				{
					Require(ShapeUtils::IsShape<VectorShape>(elementShapes.at(0)));

					const auto vectorShape = ShapeUtils::GetShape<VectorShape>(elementShapes.at(0));
					if (const auto vectorSize = ShapeUtils::GetSize<Shape::ConstantSize>(vectorShape->GetSize()))
					{
						Return(new VectorShape(new Shape::ConstantSize(listSize->GetValue() * vectorSize->GetValue())));
					}
				}
				else
				{
					auto count = 0u;
					for (const auto& elementShape : elementShapes)
					{
						Require(ShapeUtils::IsShape<VectorShape>(elementShape));

						const auto vectorShape = ShapeUtils::GetShape<VectorShape>(elementShape);
						if (const auto vectorSize = ShapeUtils::GetSize<Shape::ConstantSize>(vectorShape->GetSize()))
						{
							count += vectorSize->GetValue();
						}
						else
						{
							Return(new VectorShape(new Shape::DynamicSize(m_call)));
						}
					}
					Return(new VectorShape(new Shape::ConstantSize(count)));
				}
			}
			else
			{
				// Special case for razing a reduction

				const auto mergedCellShape = ShapeUtils::MergeShapes(listShape->GetElementShapes());
				Require(ShapeUtils::IsShape<VectorShape>(mergedCellShape));

				const auto vectorShape = ShapeUtils::GetShape<VectorShape>(mergedCellShape);
				if (const auto vectorSize = ShapeUtils::GetSize<Shape::ConstantSize>(vectorShape->GetSize()))
				{
					if (vectorSize->GetValue() == 1)
					{
						Return(new VectorShape(listShape->GetListSize()));
					}
				}
			}
			Return(new VectorShape(new Shape::DynamicSize(m_call)));
		}
		case BuiltinFunction::Primitive::List:
		{
			// -- Static elements
			// Input: Shape1*, ..., ShapeN*
			// Output: List<N, {Shape1*, ..., ShapeN*}>

			Return(new ListShape(new Shape::ConstantSize(arguments.size()), argumentShapes));
		}
		case BuiltinFunction::Primitive::ToList:
		{
			// -- Expand vector to list
			// Input: Vector<Size*>
			// Output: List<Size*, {Vector<1>}>

			const auto argumentShape = argumentShapes.at(0);
			Require(ShapeUtils::IsShape<VectorShape>(argumentShape));

			const auto vectorShape = ShapeUtils::GetShape<VectorShape>(argumentShape);
			Return(new ListShape(vectorShape->GetSize(), {new VectorShape(new Shape::ConstantSize(1))}));
		}
		//TODO: Each functions pass argument data to internal function
		case BuiltinFunction::Primitive::Each:
		{
			// -- Function call
			// Input: f, List<Size*, {Shape*}>
			// Output: List<Size*, f(Shape*)>

			const auto type = arguments.at(0)->GetType();
			const auto function = TypeUtils::GetType<FunctionType>(type)->GetFunctionDeclaration();

			const auto argumentShape = argumentShapes.at(1);
			Require(ShapeUtils::IsShape<ListShape>(argumentShape));

			const auto listShape = ShapeUtils::GetShape<ListShape>(argumentShape);

			std::vector<const Shape *> newElementShapes;
			std::vector<const Shape *> newWriteShapes;
			for (const auto& elementShape : listShape->GetElementShapes())
			{
				const auto [shapes, writeShapes] = AnalyzeCall(function, {elementShape}, {});
				Require(ShapeUtils::IsSingleShape(shapes));
				Require(ShapeUtils::IsSingleShape(writeShapes));
				newElementShapes.push_back(ShapeUtils::GetSingleShape(shapes));
				newWriteShapes.push_back(ShapeUtils::GetSingleShape(writeShapes));
			}
			return {{new ListShape(listShape->GetListSize(), newElementShapes)}, {new ListShape(listShape->GetListSize(), newWriteShapes)}};
		}
		case BuiltinFunction::Primitive::EachItem:
		{
			// -- Function call
			// Input: f, List<Size*, {Shape1*}>, List<Size*, {Shape2*}>
			// Output: List<Size*, f(Shape1*, Shape2*)>

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

			auto elementCount1 = elementShapes1.size();
			auto elementCount2 = elementShapes1.size();
			Require(elementCount1 == elementCount2 || elementCount1 == 1 || elementCount2 == 1);

			auto count = std::max(elementCount1, elementCount2);
			std::vector<const Shape *> newElementShapes;
			std::vector<const Shape *> newWriteShapes;
			for (auto i = 0u; i < count; ++i)
			{
				const auto l_inputShape1 = elementShapes1.at((elementCount1 == 1) ? 0 : i);
				const auto l_inputShape2 = elementShapes2.at((elementCount2 == 1) ? 0 : i);

				const auto [shapes, writeShapes] = AnalyzeCall(function, {l_inputShape1, l_inputShape2}, {});
				Require(ShapeUtils::IsSingleShape(shapes));
				Require(ShapeUtils::IsSingleShape(writeShapes));
				newElementShapes.push_back(ShapeUtils::GetSingleShape(shapes));
				newWriteShapes.push_back(ShapeUtils::GetSingleShape(writeShapes));
			}
			return {{new ListShape(listShape1->GetListSize(), newElementShapes)}, {new ListShape(listShape1->GetListSize(), newWriteShapes)}};
		}
		case BuiltinFunction::Primitive::EachLeft:
		{
			// -- Function call
			// Input: f, List<Size*, {Shape1*}>, Shape2*
			// Output: List<Size*, {f(Shape1*, Shape2*)}>

			const auto type = arguments.at(0)->GetType();
			const auto function = TypeUtils::GetType<FunctionType>(type)->GetFunctionDeclaration();

			const auto argumentShape1 = argumentShapes.at(1);
			const auto argumentShape2 = argumentShapes.at(2);
			Require(ShapeUtils::IsShape<ListShape>(argumentShape1));

			const auto listShape1 = ShapeUtils::GetShape<ListShape>(argumentShape1);

			std::vector<const Shape *> newElementShapes;
			std::vector<const Shape *> newWriteShapes;
			for (const auto& elementShape1 : listShape1->GetElementShapes())
			{
				const auto [shapes, writeShapes] = AnalyzeCall(function, {elementShape1, argumentShape2}, {});
				Require(ShapeUtils::IsSingleShape(shapes));
				Require(ShapeUtils::IsSingleShape(writeShapes));
				newElementShapes.push_back(ShapeUtils::GetSingleShape(shapes));
				newWriteShapes.push_back(ShapeUtils::GetSingleShape(writeShapes));
			}
			return {{new ListShape(listShape1->GetListSize(), newElementShapes)}, {new ListShape(listShape1->GetListSize(), newWriteShapes)}};
		}
		case BuiltinFunction::Primitive::EachRight:
		{
			// -- Function call
			// Input: f, Shape1*, List<Size*, {Shape2*}>
			// Output: List<Size*, {f(Shape1*, Shape2*)}>

			const auto type = arguments.at(0)->GetType();
			const auto function = TypeUtils::GetType<FunctionType>(type)->GetFunctionDeclaration();

			const auto argumentShape1 = argumentShapes.at(1);
			const auto argumentShape2 = argumentShapes.at(2);
			Require(ShapeUtils::IsShape<ListShape>(argumentShape2));

			const auto listShape2 = ShapeUtils::GetShape<ListShape>(argumentShape2);

			std::vector<const Shape *> newElementShapes;
			std::vector<const Shape *> newWriteShapes;
			for (const auto& elementShape2 : listShape2->GetElementShapes())
			{
				const auto [shapes, writeShapes] = AnalyzeCall(function, {argumentShape1, elementShape2}, {});
				Require(ShapeUtils::IsSingleShape(shapes));
				Require(ShapeUtils::IsSingleShape(writeShapes));
				newElementShapes.push_back(ShapeUtils::GetSingleShape(shapes));
				newWriteShapes.push_back(ShapeUtils::GetSingleShape(writeShapes));
			}
			return {{new ListShape(listShape2->GetListSize(), newElementShapes)}, {new ListShape(listShape2->GetListSize(), newWriteShapes)}};
		}
		case BuiltinFunction::Primitive::Match:
		{
			// -- Any to scalar
			// Input: Shape1*, Shape2*
			// Output: Vector<1>

			Return(new VectorShape(new Shape::ConstantSize(1)));
		}

		// Database
		case BuiltinFunction::Primitive::Enum:
		{
			// -- Vector enum
			// Input: Vector<Size1*> Vector<Size2*>
			// Output: Enum<Vector<Size1*>, Vector<Size2*>>
			//
			// -- List enum
			// Input: List<Size1a*, {Vector<Size1b*>}>
			// 	  List<Size2a*, {Vector<Size2b*>}>
			// Output: Enum<List1, List2>

			const auto argumentShape1 = argumentShapes.at(0);
			const auto argumentShape2 = argumentShapes.at(1);
			Require(
				(ShapeUtils::IsShape<VectorShape>(argumentShape1) && ShapeUtils::IsShape<VectorShape>(argumentShape2)) ||
				(ShapeUtils::IsShape<ListShape>(argumentShape1) && ShapeUtils::IsShape<ListShape>(argumentShape2))
			);

			if (const auto listShape1 = ShapeUtils::GetShape<ListShape>(argumentShape1))
			{
				const auto listShape2 = ShapeUtils::GetShape<ListShape>(argumentShape2);
				Require(CheckStaticEquality(listShape1->GetListSize(), listShape2->GetListSize()));

				Require(CheckStaticTabular(listShape1));
				Require(CheckStaticTabular(listShape2));
			}
			Return(new EnumerationShape(argumentShape1, argumentShape2));
		}
		case BuiltinFunction::Primitive::Dictionary:
		{
			// -- Size equality 2 arguments
			// Input: Vector<Size*> // List<Size*, {Shape*}> // Enum<Size*> // Table<Size*, Size*> // KTable<TableShape1, TableShape2>
			//    
			//    Size(Vector<Size*>) = Size*
			//    Size(List<Size*, {Shape*}> = Size*
			//    Size(Enum<Size*>) = 1
			//    Size(Table<Size2*, Size2*>) = 1
			//    Size(KTable<TableShape1, TableShape2>) = 1
			//
			//    Require: Size(Argument1) == Size(Argument2)

			const auto argumentShape1 = argumentShapes.at(0);
			const auto argumentShape2 = argumentShapes.at(1);

			const auto size1 = ShapeUtils::GetMappingSize(argumentShape1);
			const auto size2 = ShapeUtils::GetMappingSize(argumentShape2);
			Require(CheckStaticEquality(size1, size2));
			Return(new DictionaryShape(argumentShape1, argumentShape2));
		}
		case BuiltinFunction::Primitive::Table:
		{
			// -- Create table type (assume the list is well formed, i.e. Size1* vector shapes)
			// Input: Vector<Size1*>, List<Size1*, {Vector<Size2*>, ..., Vector<Size2*>}>
			// Output: Table<Size1*, Size2*>

			const auto argumentShape1 = argumentShapes.at(0);
			const auto argumentShape2 = argumentShapes.at(1);
			Require(ShapeUtils::IsShape<VectorShape>(argumentShape1));
			Require(ShapeUtils::IsShape<ListShape>(argumentShape2));

			const auto vectorSize1 = ShapeUtils::GetShape<VectorShape>(argumentShape1)->GetSize();
			const auto listShape2 = ShapeUtils::GetShape<ListShape>(argumentShape2);
			Require(CheckStaticEquality(vectorSize1, listShape2->GetListSize()));
			Require(CheckStaticTabular(listShape2));

			auto rowShape = ShapeUtils::MergeShapes(listShape2->GetElementShapes());
			Require(ShapeUtils::IsShape<VectorShape>(rowShape));

			auto rowVector = ShapeUtils::GetShape<VectorShape>(rowShape);
			Return(new TableShape(vectorSize1, rowVector->GetSize()));
		}
		case BuiltinFunction::Primitive::KeyedTable:
		{
			// -- Create compound type
			// Input: TableShape1, TableShape2
			// Output: KTable<TableShape1, TableShape2>

			const auto argumentShape1 = argumentShapes.at(0);
			const auto argumentShape2 = argumentShapes.at(1);
			Require(ShapeUtils::IsShape<TableShape>(argumentShape1));
			Require(ShapeUtils::IsShape<TableShape>(argumentShape2));
			Return(new KeyedTableShape(ShapeUtils::GetShape<TableShape>(argumentShape1), ShapeUtils::GetShape<TableShape>(argumentShape2)));
		}
		case BuiltinFunction::Primitive::Keys:
		{
			// -- Dictionary
			// Input: Dictionary<Vector1*, Shape2*>
			// Output: Vector1*
			//
			// Input: Dictionary<List1*, Shape2*>
			// Output: List1*
			//
			// Input: Dictionary<Shape1*, Shape2*>
			// Output: List<SizeDyamic, {Shape1*}>
			//
			// -- Table
			// Input: Table<Size1*, Size2*>
			// Output: Vector<Size1*>
			//
			// -- KeyedTable
			// Input: KTable<Table1*, Table2*>
			// Output: Table1*
			//
			// -- Enumeration
			// Input: Enum<Shape1*, Shape2*>
			// Output: Shape1*

			const auto argumentShape = argumentShapes.at(0);
			if (const auto dictionaryShape = ShapeUtils::GetShape<DictionaryShape>(argumentShape))
			{
				const auto keyShape = dictionaryShape->GetKeyShape();
				if (ShapeUtils::IsShape<VectorShape>(keyShape) || ShapeUtils::IsShape<ListShape>(keyShape))
				{
					Return(keyShape);
				}
				Return(new ListShape(new Shape::DynamicSize(m_call), {keyShape}));
			}
			else if (const auto tableShape = ShapeUtils::GetShape<TableShape>(argumentShape))
			{
				Return(new VectorShape(tableShape->GetColumnsSize()));
			}
			else if (const auto kTableShape = ShapeUtils::GetShape<KeyedTableShape>(argumentShape))
			{
				Return(kTableShape->GetKeyShape());
			}
			else if (const auto enumShape = ShapeUtils::GetShape<EnumerationShape>(argumentShape))
			{
				Return(enumShape->GetKeyShape());
			}
			break;
		}
		case BuiltinFunction::Primitive::Values:
		{
			// -- Dictionary
			// Input: Dictionary<Shape1*, List2*>
			// Output: List2*
			//
			// Input: Dictionary<Shape1*, Shape2*>
			// Output: List<SizeDynamic, {Shape2*}>
			//
			// -- Table
			// Input: Table<Size1*, Size2*>
			// Output: List<Size1*, {Vector<Size2*>}>
			//
			// -- KeyedTable
			// Input: KTable<Table1*, Table2*>
			// Output: Table2*
			//
			// -- Enumeration
			// Input: Enum<Shape1*, Shape2*>
			// Output: Vector<size(Shape2*)>

			const auto argumentShape = argumentShapes.at(0);
			if (const auto dictionaryShape = ShapeUtils::GetShape<DictionaryShape>(argumentShape))
			{
				const auto valueShape = dictionaryShape->GetValueShape();
				if (ShapeUtils::IsShape<ListShape>(valueShape))
				{
					Return(valueShape);
				}
				Return(new ListShape(new Shape::DynamicSize(m_call), {valueShape}));
			}
			else if (const auto tableShape = ShapeUtils::GetShape<TableShape>(argumentShape))
			{
				const auto colsSize = tableShape->GetColumnsSize();
				const auto rowsSize = tableShape->GetRowsSize();
				Return(new ListShape(colsSize, {new VectorShape(rowsSize)}));
			}
			else if (const auto kTableShape = ShapeUtils::GetShape<KeyedTableShape>(argumentShape))
			{
				Return(kTableShape->GetValueShape());
			}
			else if (const auto enumShape = ShapeUtils::GetShape<EnumerationShape>(argumentShape))
			{
				// Returns index vector

				auto valueShape = enumShape->GetValueShape();
				if (const auto vectorValueShape = ShapeUtils::GetShape<VectorShape>(valueShape))
				{
					Return(vectorValueShape);
				}
				else if (const auto listValueShape = ShapeUtils::GetShape<ListShape>(valueShape))
				{
					auto cellShape = ShapeUtils::MergeShapes(listValueShape->GetElementShapes());
					if (const auto vectorCellShape = ShapeUtils::GetShape<VectorShape>(cellShape))
					{
						Return(vectorCellShape);
					}
				}
			}
			break;
		}
		case BuiltinFunction::Primitive::Meta:
		{
			// -- Always dynamic
			// Input: Table* || KeyedTable*
			// Output: Table<SizeDynamic1, SizeDynamic2>

			const auto argumentShape = argumentShapes.at(0);
			Require(ShapeUtils::IsShape<TableShape>(argumentShape) || ShapeUtils::IsShape<KeyedTableShape>(argumentShape));
			Return(new TableShape(new Shape::DynamicSize(m_call, 1), new Shape::DynamicSize(m_call, 2)));
		}
		case BuiltinFunction::Primitive::Fetch:
		{
			// -- Enum unpacking (get the value)
			// Input: Enum<Shape1*, Shape2*>
			// Output: Shape2*

			const auto argumentShape = argumentShapes.at(0);
			Require(ShapeUtils::IsShape<EnumerationShape>(argumentShape));
			Return(ShapeUtils::GetShape<EnumerationShape>(argumentShape)->GetValueShape());
		}
		case BuiltinFunction::Primitive::ColumnValue:
		{
			// -- Always dynamic
			// Input: Table<Size1*, Size2*>, Vector<1>
			// Output: Vector<Size2*> (rows)

			const auto argumentShape1 = argumentShapes.at(0);
			const auto argumentShape2 = argumentShapes.at(1);
			Require(ShapeUtils::IsShape<TableShape>(argumentShape1));
			Require(ShapeUtils::IsShape<VectorShape>(argumentShape2));

			const auto vectorShape2 = ShapeUtils::GetShape<VectorShape>(argumentShape2);
			Require(CheckStaticScalar(vectorShape2->GetSize()));

			const auto tableShape = ShapeUtils::GetShape<TableShape>(argumentShape1);
			const auto rowsSize = tableShape->GetRowsSize();

			if (const auto [isConstant, value] = GetConstantArgument<const SymbolValue *>(arguments, 1); isConstant)
			{
				// Foreign keys for TPC-H

				auto columnName = value->GetName();
				if (columnName == "n_regionkey")
				{
					Return(new EnumerationShape(new VectorShape(new Shape::SymbolSize("data.region.rows")), new VectorShape(rowsSize)));
				}
				else if (columnName == "s_nationkey" || columnName == "c_nationkey")
				{
					Return(new EnumerationShape(new VectorShape(new Shape::SymbolSize("data.customer.rows")), new VectorShape(rowsSize)));
				}
				else if (columnName == "ps_suppkey")
				{
					Return(new EnumerationShape(new VectorShape(new Shape::SymbolSize("data.supplier.rows")), new VectorShape(rowsSize)));
				}
				else if (columnName == "ps_partkey")
				{
					Return(new EnumerationShape(new VectorShape(new Shape::SymbolSize("data.part.rows")), new VectorShape(rowsSize)));
				}
				else if (columnName == "o_custkey")
				{
					Return(new EnumerationShape(new VectorShape(new Shape::SymbolSize("data.customer.rows")), new VectorShape(rowsSize)));
				}
				else if (columnName == "l_orderkey")
				{
					Return(new EnumerationShape(new VectorShape(new Shape::SymbolSize("data.orders.rows")), new VectorShape(rowsSize)));
				}
				// else if (columnName == "l_partkey" || columnName == "l_suppkey")
				// {
				// 	Return(new EnumerationShape(new VectorShape(new Shape::SymbolSize("data.partsupp.rows")), new VectorShape(rowsSize)));
				// }
				else if (columnName.rfind("enum_", 0) == 0)
				{
					// Debug foreign key is in the same table

					Return(new EnumerationShape(new VectorShape(rowsSize), new VectorShape(rowsSize)));
				}
			}
			Return(new VectorShape(rowsSize));
		}
		case BuiltinFunction::Primitive::LoadTable:
		{
			// -- Static table rows/cols
			// Input: Vector<1> (table k)
			// Output: Table<SizeSymbol<k.cols>, SizeSymbol<k.rows>>
			//
			// -- Dynamic table
			// Input: Vector<1>
			// Output: Table<SizeDynamic1, SizeDynamic2>

			const auto argumentShape = argumentShapes.at(0);
			Require(ShapeUtils::IsShape<VectorShape>(argumentShape));

			const auto vectorShape = ShapeUtils::GetShape<VectorShape>(argumentShape);
			Require(CheckStaticScalar(vectorShape->GetSize()));

			if (const auto [isConstant, value] = GetConstantArgument<const SymbolValue *>(arguments, 0); isConstant)
			{
				auto tableName = value->GetName();
				Return(new TableShape(new Shape::SymbolSize("data." + tableName + ".cols"), new Shape::SymbolSize("data." + tableName + ".rows")));
			}
			Return(new TableShape(new Shape::DynamicSize(m_call, 1), new Shape::DynamicSize(m_call, 2)));
		}
		case BuiltinFunction::Primitive::JoinIndex:
		{
			Require(AnalyzeJoinArguments(argumentShapes, arguments));
			Return(new ListShape(new Shape::ConstantSize(2), {new VectorShape(new Shape::DynamicSize(m_call))}));
		}

		// Indexing
		case BuiltinFunction::Primitive::Index:
		{
			// -- Right propagation
			// Input: Vector<Size1*>, Vector<Size2*>
			// Output: Vector<Size2*>
			//
			// Input: List<Size1*, {Shape2*}>, Vector<Size2*>
			// Output: Shape2*

			const auto argumentShape1 = argumentShapes.at(0);
			const auto argumentShape2 = argumentShapes.at(1);
			Require(ShapeUtils::IsShape<VectorShape>(argumentShape2));

			if (ShapeUtils::IsShape<VectorShape>(argumentShape1))
			{
				Return(argumentShape2);
			}
			else if (const auto listShape1 = ShapeUtils::GetShape<ListShape>(argumentShape1))
			{
				auto vectorShape2 = ShapeUtils::GetShape<VectorShape>(argumentShape2);
				Require(CheckStaticScalar(vectorShape2->GetSize()));

				const auto& elementShapes = listShape1->GetElementShapes();
				if (elementShapes.size() == 1)
				{
					Return(elementShapes.at(0));
				}

				if (const auto [isConstant, value] = GetConstantArgument<std::int64_t>(arguments, 1); isConstant)
				{
					if (value < elementShapes.size())
					{
						Return(elementShapes.at(value));
					}
					break;
				}
				Return(ShapeUtils::MergeShapes(elementShapes));
			}
			break;
		}
		case BuiltinFunction::Primitive::IndexAssignment:
		{
			// -- Left propagation
			// Input: Vector<Size1*>, Vector<Size2*>, Vector<Size2*>
			// Output: Vector<Size1*>

			const auto argumentShape1 = argumentShapes.at(0);
			const auto argumentShape2 = argumentShapes.at(1);
			const auto argumentShape3 = argumentShapes.at(2);
			Require(ShapeUtils::IsShape<VectorShape>(argumentShape1));
			Require(ShapeUtils::IsShape<VectorShape>(argumentShape2));
			Require(ShapeUtils::IsShape<VectorShape>(argumentShape3));

			const auto vectorShape2 = ShapeUtils::GetShape<VectorShape>(argumentShape2);
			const auto vectorShape3 = ShapeUtils::GetShape<VectorShape>(argumentShape3);
			Require(CheckStaticEquality(vectorShape2->GetSize(), vectorShape2->GetSize()));

			// Write shape differs from active shape
			return {{argumentShape1}, {argumentShape2}};
		}

		// Other
		case BuiltinFunction::Primitive::LoadCSV:
		{
			// -- Dynamic table load
			// Input: Vector<1>
			// Output: Table<SizeDynamic1, SizeDynamic2>

			const auto argumentShape = argumentShapes.at(0);
			Require(ShapeUtils::IsShape<VectorShape>(argumentShape));

			const auto vectorShape = ShapeUtils::GetShape<VectorShape>(argumentShape);
			Require(CheckStaticScalar(vectorShape->GetSize()));
			Return(new TableShape(new Shape::DynamicSize(m_call, 1), new Shape::DynamicSize(m_call, 2)));
		}
		case BuiltinFunction::Primitive::Print:
		case BuiltinFunction::Primitive::String:
		{
			// -- Any shape
			// Input: Shape*
			// Outout: Vector<1>

			Return(new VectorShape(new Shape::ConstantSize(1)));
		}
		case BuiltinFunction::Primitive::SubString:
		{
			// -- Substring of vector
			// Input: Vector<Size*>, Vector<2>
			// Output: Vector<Size*>

			const auto argumentShape1 = argumentShapes.at(0);
			const auto argumentShape2 = argumentShapes.at(1);
			Require(ShapeUtils::IsShape<VectorShape>(argumentShape1));
			Require(ShapeUtils::IsShape<VectorShape>(argumentShape2));
			Return(argumentShape1);
		}

		// GPU
		case BuiltinFunction::Primitive::GPUOrderLib:
		{
			// For both, optional order

			// -- Vector input
			// Input: *, *, [*,] Vector<Size*>, Vector<1>
			// Output: Vector<Size*>
			//
			// -- List input
			// Input: *, *, [*,] List<k, {Vector<Size*>}>, Vector<k>
			// Output: Vector<Size*>

			const auto isShared = TypeUtils::IsType<FunctionType>(arguments.at(2)->GetType());

			const auto initType = arguments.at(0)->GetType();
			const auto sortType = arguments.at(1)->GetType();
			const auto dataShape = argumentShapes.at(2 + isShared);

			const auto initFunction = TypeUtils::GetType<FunctionType>(initType)->GetFunctionDeclaration();
			const auto sortFunction = TypeUtils::GetType<FunctionType>(sortType)->GetFunctionDeclaration();

			Require(ShapeUtils::IsShape<VectorShape>(dataShape) || ShapeUtils::IsShape<ListShape>(dataShape));

			if (arguments.size() == (3 + isShared))
			{
				// Init call

				const auto [initShapes, initWriteShapes] = AnalyzeCall(initFunction, {dataShape}, {});
				Require(initShapes.size() == 2);
				Require(ShapeUtils::IsShape<VectorShape>(initShapes.at(0)));

				// Sort call

				const auto [sortShapes, sortWriteShapes] = AnalyzeCall(sortFunction, {initShapes.at(0), initShapes.at(1)}, {});
				Require(sortShapes.size() == 0);

				if (isShared)
				{
					const auto sharedType = arguments.at(2)->GetType();
					const auto sharedFunction = TypeUtils::GetType<FunctionType>(sharedType)->GetFunctionDeclaration();
					const auto [sharedShapes, sharedWriteShapes] = AnalyzeCall(sharedFunction, {initShapes.at(0), initShapes.at(1)}, {});
					Require(sharedShapes.size() == 0);
				}

				// Return

				if (ShapeUtils::IsShape<VectorShape>(dataShape))
				{
					Return(dataShape);
				}
				else if (const auto listShape = ShapeUtils::GetShape<ListShape>(dataShape))
				{
					Require(CheckStaticTabular(listShape));
					Return(ShapeUtils::MergeShapes(listShape->GetElementShapes()));
				}
			}

			const auto orderShape = argumentShapes.at(3 + isShared);
			Require(ShapeUtils::IsShape<VectorShape>(orderShape));

			// Init call

			const auto [initShapes, initWriteShapes] = AnalyzeCall(initFunction, {dataShape, orderShape}, {});
			Require(initShapes.size() == 2);
			Require(ShapeUtils::IsShape<VectorShape>(initShapes.at(0)));

			// Sort call

			const auto [sortShapes, sortWriteShapes] = AnalyzeCall(sortFunction, {initShapes.at(0), initShapes.at(1), orderShape}, {});
			Require(sortShapes.size() == 0);

			if (isShared)
			{
				const auto sharedType = arguments.at(2)->GetType();
				const auto sharedFunction = TypeUtils::GetType<FunctionType>(sharedType)->GetFunctionDeclaration();
				const auto [sharedShapes, sharedWriteShapes] = AnalyzeCall(sharedFunction, {initShapes.at(0), initShapes.at(1), orderShape}, {});
				Require(sharedShapes.size() == 0);
			}

			// Return

			const auto orderVector = ShapeUtils::GetShape<VectorShape>(orderShape);
			if (ShapeUtils::IsShape<VectorShape>(dataShape))
			{
				Require(CheckStaticScalar(orderVector->GetSize()));
				Return(dataShape);
			}
			else if (const auto listShape = ShapeUtils::GetShape<ListShape>(dataShape))
			{
				Require(CheckStaticEquality(listShape->GetListSize(), orderVector->GetSize()));
				Require(CheckStaticTabular(listShape));
				Return(ShapeUtils::MergeShapes(listShape->GetElementShapes()));
			}
			break;
		}
		case BuiltinFunction::Primitive::GPUOrderInit:
		{
			// -- Vector input
			// Input: Vector<Size*>, Vector<1>
			// Output: Vector<Power2(Size*)>, Vector<Power2(Size*)>
			//
			// -- List input
			// Input: List<k, {Vector<Size*>}>, Vector<k>
			// Output: Vector<Power2(Size*)>, List<k, {Vector<Power2(Size*)>}>

			const auto argumentShape0 = argumentShapes.at(0);
			const auto argumentShape1 = argumentShapes.at(1);

			Require(ShapeUtils::IsShape<VectorShape>(argumentShape1));
			const auto vectorShape1 = ShapeUtils::GetShape<VectorShape>(argumentShape1);

			if (const auto vectorShape0 = ShapeUtils::GetShape<VectorShape>(argumentShape0))
			{
				Require(CheckStaticScalar(vectorShape1->GetSize()));
				if (const auto constantSize = ShapeUtils::GetSize<Shape::ConstantSize>(vectorShape0->GetSize()))
				{
					auto powerSize = Utils::Math::Power2(constantSize->GetValue());
					//TODO: Is this needed?
					if (powerSize < 2048)
					{
						powerSize = 2048;
					}
					const auto indexShape = new VectorShape(new Shape::ConstantSize(powerSize));
					Return2(indexShape, indexShape);
				}

				const auto indexShape = new VectorShape(new Shape::DynamicSize(m_call));
				Return2(indexShape, indexShape);
			}
			else if (const auto listShape = ShapeUtils::GetShape<ListShape>(argumentShape0))
			{
				Require(CheckStaticEquality(listShape->GetListSize(), vectorShape1->GetSize()) || CheckStaticScalar(vectorShape1->GetSize()));
				Require(CheckStaticTabular(listShape));

				auto mergedShape = ShapeUtils::MergeShapes(listShape->GetElementShapes());
				if (const auto vectorShape = ShapeUtils::GetShape<VectorShape>(mergedShape))
				{
					if (const auto constantSize = ShapeUtils::GetSize<Shape::ConstantSize>(vectorShape->GetSize()))
					{
						auto powerSize = Utils::Math::Power2(constantSize->GetValue());
						if (powerSize < 2048)
						{
							powerSize = 2048;
						}
						const auto indexShape = new VectorShape(new Shape::ConstantSize(powerSize));
						Return2(indexShape, new ListShape(listShape->GetListSize(), {indexShape}));
					}

					const auto indexShape = new VectorShape(new Shape::DynamicSize(m_call));
					Return2(indexShape, new ListShape(listShape->GetListSize(), {indexShape}));
				}
			}
			break;
		}
		case BuiltinFunction::Primitive::GPUOrder:
		case BuiltinFunction::Primitive::GPUOrderShared:
		{
			// -- Vector input
			// Input: Vector<Power2(Size*)>, Vector<Power2(Size*)>, Vector<1>
			// Output: -
			//
			// -- List input
			// Input: Vector<Power2(Size*)>, List<k, {Vector<Power2(Size*)>}>, Vector<k>
			// Output: -

			const auto argumentShape0 = argumentShapes.at(0);
			const auto argumentShape1 = argumentShapes.at(1);
			const auto argumentShape2 = argumentShapes.at(2);

			Require(ShapeUtils::IsShape<VectorShape>(argumentShape0));
			const auto vectorShape0 = ShapeUtils::GetShape<VectorShape>(argumentShape0);

			if (const auto constantSize = ShapeUtils::GetSize<Shape::ConstantSize>(vectorShape0->GetSize()))
			{
				auto value = constantSize->GetValue();
				Require(value == Utils::Math::Power2(value));
			}

			Require(ShapeUtils::IsShape<VectorShape>(argumentShape2));
			const auto vectorShape2 = ShapeUtils::GetShape<VectorShape>(argumentShape2);

			if (const auto vectorShape1 = ShapeUtils::GetShape<VectorShape>(argumentShape1))
			{
				Require(CheckStaticScalar(vectorShape2->GetSize()));
				Require(CheckStaticEquality(vectorShape0->GetSize(), vectorShape1->GetSize()));
				Return();
			}
			else if (const auto listShape = ShapeUtils::GetShape<ListShape>(argumentShape1))
			{
				Require(CheckStaticEquality(listShape->GetListSize(), vectorShape2->GetSize()) || CheckStaticScalar(vectorShape2->GetSize()));
				Require(CheckStaticTabular(listShape));

				auto mergedShape = ShapeUtils::MergeShapes(listShape->GetElementShapes());
				if (const auto vectorShape = ShapeUtils::GetShape<VectorShape>(mergedShape))
				{
					Require(CheckStaticEquality(vectorShape0->GetSize(), vectorShape->GetSize()));
					Return();
				}
			}
			break;
		}
		case BuiltinFunction::Primitive::GPUGroupLib:
		{
			// -- Vector/list group
			// Input: *, *, [*,] *, {Vector<Size*> | List<Size*, {Shape*}>}
			// Output: Dictionary<Vector<SizeDynamic1>, Vector<SizeDynamic2>>
			//
			// For lists, ensure all shapes are vectors of the same length

			const auto isShared = (arguments.size() == 5);

			const auto initType = arguments.at(0)->GetType();
			const auto sortType = arguments.at(1)->GetType();
			const auto groupType = arguments.at(2 + isShared)->GetType();

			const auto dataShape = argumentShapes.at(3 + isShared);
			Require(ShapeUtils::IsShape<VectorShape>(dataShape) || ShapeUtils::IsShape<ListShape>(dataShape));

			// Fetch functions

			const auto initFunction = TypeUtils::GetType<FunctionType>(initType)->GetFunctionDeclaration();
			const auto sortFunction = TypeUtils::GetType<FunctionType>(sortType)->GetFunctionDeclaration();
			const auto groupFunction = TypeUtils::GetType<FunctionType>(groupType)->GetFunctionDeclaration();

			// Init call

			const auto [initShapes, initWriteShapes] = AnalyzeCall(initFunction, {dataShape}, {});
			Require(initShapes.size() == 2);
			Require(ShapeUtils::IsShape<VectorShape>(initShapes.at(0)));

			// Sort call

			const auto [sortShapes, sortWriteShapes] = AnalyzeCall(sortFunction, initShapes, {});
			Require(sortShapes.size() == 0);

			// Sort shared call

			if (isShared)
			{
				const auto sharedType = arguments.at(2)->GetType();
				const auto sharedFunction = TypeUtils::GetType<FunctionType>(sharedType)->GetFunctionDeclaration();

				const auto [sharedShapes, sharedWriteShapes] = AnalyzeCall(sharedFunction, initShapes, {});
				Require(sharedShapes.size() == 0);
			}

			// Group call

			const VectorShape *indexShape = nullptr;
			if (const auto vectorShape = ShapeUtils::GetShape<VectorShape>(dataShape))
			{
				indexShape = vectorShape;
			}
			else if (const auto listShape = ShapeUtils::GetShape<ListShape>(dataShape))
			{
				const auto mergedShape = ShapeUtils::MergeShapes(listShape->GetElementShapes());
				Require(ShapeUtils::IsShape<VectorShape>(mergedShape));
				indexShape = ShapeUtils::GetShape<VectorShape>(mergedShape);
			}

			const auto [groupShapes, groupWriteShapes] = AnalyzeCall(groupFunction, {indexShape, dataShape}, {});
			Require(groupShapes.size() == 2);
			Require(ShapeUtils::IsShape<VectorShape>(groupShapes.at(0)));
			Require(ShapeUtils::IsShape<VectorShape>(groupShapes.at(1)));

			Return(new DictionaryShape(
				new VectorShape(new Shape::DynamicSize(m_call, 1)),
				new VectorShape(new Shape::DynamicSize(m_call, 2))
			));
		}
		case BuiltinFunction::Primitive::GPUGroup:
		{
			// -- Vector/list group
			// Input: Vector<Size*>, {Vector<Size*> | List<Size2*, {Vector<Size*>}>}
			// Output: Vector<DynamicSize1>, Vector<DynamicSize2>
			//
			// For lists, ensure all shapes are vectors of the same length

			const auto argumentShape0 = argumentShapes.at(0);
			const auto argumentShape1 = argumentShapes.at(1);

			Require(ShapeUtils::IsShape<VectorShape>(argumentShape0));
			const auto vectorShape0 = ShapeUtils::GetShape<VectorShape>(argumentShape0);

			if (const auto vectorShape1 = ShapeUtils::GetShape<VectorShape>(argumentShape1))
			{
				Require(CheckStaticEquality(vectorShape0->GetSize(), vectorShape1->GetSize()));
			}
			else if (const auto listShape = ShapeUtils::GetShape<ListShape>(argumentShape1))
			{
				Require(CheckStaticTabular(listShape));

				auto mergedShape = ShapeUtils::MergeShapes(listShape->GetElementShapes());
				if (const auto vectorShape = ShapeUtils::GetShape<VectorShape>(mergedShape))
				{
					Require(CheckStaticEquality(vectorShape0->GetSize(), vectorShape->GetSize()));
				}
			}

			const auto returnShape1 = new VectorShape(new Shape::CompressedSize(new DataObject(), vectorShape0->GetSize()));
			const auto returnShape2 = new VectorShape(new Shape::CompressedSize(new DataObject(), vectorShape0->GetSize()));
			Return2(returnShape1, returnShape2);
		}
		case BuiltinFunction::Primitive::GPUUniqueLib:
		{
			// -- Vector/list group
			// Input: *, *, [*,] *, Vector<Size*>
			// Output: Vector<DynamicSize1>

			const auto isShared = (arguments.size() == 5);

			const auto initType = arguments.at(0)->GetType();
			const auto sortType = arguments.at(1)->GetType();
			const auto uniqueType = arguments.at(2 + isShared)->GetType();

			const auto dataShape = argumentShapes.at(3 + isShared);
			Require(ShapeUtils::IsShape<VectorShape>(dataShape));

			// Fetch functions

			const auto initFunction = TypeUtils::GetType<FunctionType>(initType)->GetFunctionDeclaration();
			const auto sortFunction = TypeUtils::GetType<FunctionType>(sortType)->GetFunctionDeclaration();
			const auto uniqueFunction = TypeUtils::GetType<FunctionType>(uniqueType)->GetFunctionDeclaration();

			// Init call

			const auto [initShapes, initWriteShapes] = AnalyzeCall(initFunction, {dataShape}, {});
			Require(initShapes.size() == 2);
			Require(ShapeUtils::IsShape<VectorShape>(initShapes.at(0)));

			// Sort call

			const auto [sortShapes, sortWriteShapes] = AnalyzeCall(sortFunction, initShapes, {});
			Require(sortShapes.size() == 0);

			// Sort call

			if (isShared)
			{
				const auto sharedType = arguments.at(2)->GetType();
				const auto sharedFunction = TypeUtils::GetType<FunctionType>(sharedType)->GetFunctionDeclaration();
				const auto [sharedShapes, sharedWriteShapes] = AnalyzeCall(sharedFunction, initShapes, {});
				Require(sharedShapes.size() == 0);
			}

			// Unique call

			const auto [uniqueShapes, uniqueWriteShapes] = AnalyzeCall(uniqueFunction, {dataShape, dataShape}, {});
			Require(uniqueShapes.size() == 1);
			Require(ShapeUtils::IsShape<VectorShape>(uniqueShapes.at(0)));

			Return(uniqueShapes.at(0));
		}
		case BuiltinFunction::Primitive::GPUUnique:
		{
			// -- Vector/list group
			// Input: Vector<Size*>, Vector<Size*>
			// Output: Vector<DynamicSize1>

			const auto argumentShape0 = argumentShapes.at(0);
			const auto argumentShape1 = argumentShapes.at(1);

			Require(ShapeUtils::IsShape<VectorShape>(argumentShape0));
			Require(ShapeUtils::IsShape<VectorShape>(argumentShape1));

			const auto vectorShape0 = ShapeUtils::GetShape<VectorShape>(argumentShape0);
			const auto vectorShape1 = ShapeUtils::GetShape<VectorShape>(argumentShape1);
			Require(CheckStaticEquality(vectorShape0->GetSize(), vectorShape1->GetSize()));

			Return(new VectorShape(new Shape::CompressedSize(new DataObject(), vectorShape0->GetSize())));
		}
		case BuiltinFunction::Primitive::GPULoopJoinLib:
		{
			// -- Vector input
			// Input: *, *, Vector<Size*>, Vector<Size*>
			// Output: List<2, {Vector<SizeDynamic/m>}>
			//
			// -- List input
			// Input: *, *, List<k, {Vector<Size*>}>, List<k, {Vector<Size*>}>
			// Output: List<2, {Vector<SizeDynamic/m>}>

			const auto countType = arguments.at(0)->GetType();
			const auto joinType = arguments.at(1)->GetType();

			const auto countFunction = TypeUtils::GetType<FunctionType>(countType)->GetFunctionDeclaration();
			const auto joinFunction = TypeUtils::GetType<FunctionType>(joinType)->GetFunctionDeclaration();

			const auto argumentShape3 = argumentShapes.at(2);
			const auto argumentShape4 = argumentShapes.at(3);

			Require(ShapeUtils::IsShape<VectorShape>(argumentShape3) || ShapeUtils::IsShape<ListShape>(argumentShape3));
			Require(ShapeUtils::IsShape<VectorShape>(argumentShape4) || ShapeUtils::IsShape<ListShape>(argumentShape4));

			// Count call

			const auto [countShapes, countWriteShapes] = AnalyzeCall(countFunction, {argumentShape3, argumentShape4}, {});
			Require(countShapes.size() == 2);
			Require(ShapeUtils::IsShape<VectorShape>(countShapes.at(0)));
			Require(ShapeUtils::IsShape<VectorShape>(countShapes.at(1)));

			const auto vectorOffsets = ShapeUtils::GetShape<VectorShape>(countShapes.at(0));
			const auto vectorCount = ShapeUtils::GetShape<VectorShape>(countShapes.at(1));
			Require(CheckStaticScalar(vectorCount->GetSize()));

			if (const auto vectorRight = ShapeUtils::GetShape<VectorShape>(argumentShape4))
			{
				Require(CheckStaticEquality(vectorOffsets->GetSize(), vectorRight->GetSize()));
			}
			else if (const auto listRight = ShapeUtils::GetShape<ListShape>(argumentShape4))
			{
				Require(CheckStaticTabular(listRight));

				auto cellShape = ShapeUtils::MergeShapes(listRight->GetElementShapes());
				Require(ShapeUtils::IsShape<VectorShape>(cellShape));

				auto vectorCell = ShapeUtils::GetShape<VectorShape>(cellShape);
				Require(CheckStaticEquality(vectorOffsets->GetSize(), vectorCell->GetSize()));
			}

			// Join call

			const auto [joinShapes, joinWriteShapes] = AnalyzeCall(joinFunction, {argumentShape3, argumentShape4, countShapes.at(0), countShapes.at(1)}, {});
			Require(joinShapes.size() == 1);
			Require(ShapeUtils::IsShape<ListShape>(joinShapes.at(0)));

			const auto indexesShape = ShapeUtils::GetShape<ListShape>(joinShapes.at(0));
			const auto indexesSize = indexesShape->GetListSize();

			Require(ShapeUtils::IsSize<Shape::ConstantSize>(indexesSize));
			Require(ShapeUtils::GetSize<Shape::ConstantSize>(indexesSize)->GetValue() == 2);
			Require(CheckStaticTabular(indexesShape));

			// Return

			return {joinShapes, joinWriteShapes};
		}
		case BuiltinFunction::Primitive::GPULoopJoinCount:
		{
			// -- Vector input
			// Input: Vector<Size*>, Vector<Size*>
			// Output: Vector<Size*>, Vector<1>
			//
			// -- List input
			// Input: List<k, {Vector<Size*>}>, List<k, {Vector<Size*>}>
			// Output: Vector<Size*>, Vector<1>

			Require(AnalyzeJoinArguments(argumentShapes, arguments));

			// Return shape is the left argument size

			auto rightShape = argumentShapes.at(argumentShapes.size() - 1);

			// Get the count shape (constant for runtime, compressed for codegen)

			const auto countShape = new VectorShape(new Shape::ConstantSize(1));
			const auto countWriteShape = new VectorShape(new Shape::CompressedSize(new DataObject(), new Shape::ConstantSize(1)));

			if (const auto vectorShape = ShapeUtils::GetShape<VectorShape>(rightShape))
			{
				std::vector<const Shape *> shapes({vectorShape, countShape});
				std::vector<const Shape *> writeShapes({vectorShape, countWriteShape});
				return {shapes, writeShapes};
			}
			else if (const auto listShape = ShapeUtils::GetShape<ListShape>(rightShape))
			{
				auto cellShape = ShapeUtils::MergeShapes(listShape->GetElementShapes());
				Require(ShapeUtils::IsShape<VectorShape>(cellShape));

				std::vector<const Shape *> shapes({cellShape, countShape});
				std::vector<const Shape *> writeShapes({cellShape, countWriteShape});
				return {shapes, writeShapes};
			}
			break;
		}
		case BuiltinFunction::Primitive::GPULoopJoin:
		{
			// -- Unknown scalar constant
			// Intput: Vector<Size*>, Vector<Size*>, Vector<Size*>, Vector<1> | List<k, {Vector<Size*>}>, List<k, {Vector<Size*>}>, Vector<Size*>, Vector<1>
			// Output: List<2, {Vector<SizeDynamic>}>
			//
			// -- Known scalar constant
			// Intput: Vector<Size*>, Vector<Size*>, Vector<Size*>, Vector<1> (value m) | List<k, {Vector<Size*>}>, List<k, {Vector<Size*>}>, Vector<Size*>, Vector<1> (value m)
			// Output: List<2, {Vector<m>}>

			std::vector<const Shape *> joinShapes(std::begin(argumentShapes), std::end(argumentShapes) - 2);
			std::vector<const Operand *> joinArguments(std::begin(arguments), std::end(arguments) - 2);

			Require(AnalyzeJoinArguments(joinShapes, joinArguments));

			const auto vectorOffsets = ShapeUtils::GetShape<VectorShape>(argumentShapes.at(argumentShapes.size() - 2));
			const auto vectorCount = ShapeUtils::GetShape<VectorShape>(argumentShapes.at(argumentShapes.size() - 1));
			Require(CheckStaticScalar(vectorCount->GetSize()));

			const auto rightShape = argumentShapes.at(argumentShapes.size() - 3);
			if (const auto vectorRight = ShapeUtils::GetShape<VectorShape>(rightShape))
			{
				Require(CheckStaticEquality(vectorOffsets->GetSize(), vectorRight->GetSize()));
			}
			else if (const auto listRight = ShapeUtils::GetShape<ListShape>(rightShape))
			{
				auto cellShape = ShapeUtils::MergeShapes(listRight->GetElementShapes());
				Require(ShapeUtils::IsShape<VectorShape>(cellShape));

				auto vectorCell = ShapeUtils::GetShape<VectorShape>(cellShape);
				Require(CheckStaticEquality(vectorOffsets->GetSize(), vectorCell->GetSize()));
			}

			if (arguments.size() > 0)
			{
				// Get the constant count parameter

				if (const auto [isConstant, value] = GetConstantArgument<std::int64_t>(arguments, arguments.size() - 1); isConstant)
				{
					Return(new ListShape(new Shape::ConstantSize(2), {new VectorShape(new Shape::ConstantSize(value))}));
				}
			}
			Return(new ListShape(new Shape::ConstantSize(2), {new VectorShape(new Shape::DynamicSize(m_call))}));
		}
		case BuiltinFunction::Primitive::GPUHashJoinLib:
		{
			// -- Vector input
			// Input: *, *, *, Vector<Size*>, Vector<Size*>
			// Output: List<2, {Vector<SizeDynamic/m>}>
			//
			// -- List input
			// Input: *, *, *, List<k, {Vector<Size*>}>, List<k, {Vector<Size*>}>
			// Output: List<2, {Vector<SizeDynamic/m>}>

			const auto hashType = arguments.at(0)->GetType();
			const auto countType = arguments.at(1)->GetType();
			const auto joinType = arguments.at(2)->GetType();

			const auto hashFunction = TypeUtils::GetType<FunctionType>(hashType)->GetFunctionDeclaration();
			const auto countFunction = TypeUtils::GetType<FunctionType>(countType)->GetFunctionDeclaration();
			const auto joinFunction = TypeUtils::GetType<FunctionType>(joinType)->GetFunctionDeclaration();

			const auto leftShape = argumentShapes.at(3);
			const auto rightShape = argumentShapes.at(4);

			Require(ShapeUtils::IsShape<VectorShape>(leftShape) || ShapeUtils::IsShape<ListShape>(leftShape));
			Require(ShapeUtils::IsShape<VectorShape>(rightShape) || ShapeUtils::IsShape<ListShape>(rightShape));

			// Hash call

			const auto [hashShapes, hashWriteShapes] = AnalyzeCall(hashFunction, {leftShape}, {});
			Require(hashShapes.size() == 2);

			if (ShapeUtils::IsShape<VectorShape>(leftShape))
			{
				Require(ShapeUtils::IsShape<VectorShape>(hashShapes.at(0)));
			}
			else if (ShapeUtils::IsShape<ListShape>(leftShape))
			{
				Require(ShapeUtils::IsShape<ListShape>(hashShapes.at(0)));
				auto listShape = ShapeUtils::GetShape<ListShape>(hashShapes.at(0));
				Require(CheckStaticTabular(listShape));
			}
			Require(ShapeUtils::IsShape<VectorShape>(hashShapes.at(1)));

			// Count call

			const auto [countShapes, countWriteShapes] = AnalyzeCall(countFunction, {hashShapes.at(0), rightShape}, {});
			Require(countShapes.size() == 2);
			Require(ShapeUtils::IsShape<VectorShape>(countShapes.at(0)));
			Require(ShapeUtils::IsShape<VectorShape>(countShapes.at(1)));

			const auto vectorOffsets = ShapeUtils::GetShape<VectorShape>(countShapes.at(0));
			const auto vectorCount = ShapeUtils::GetShape<VectorShape>(countShapes.at(1));
			Require(CheckStaticScalar(vectorCount->GetSize()));

			if (const auto vectorRight = ShapeUtils::GetShape<VectorShape>(rightShape))
			{
				Require(CheckStaticEquality(vectorOffsets->GetSize(), vectorRight->GetSize()));
			}
			else if (const auto listRight = ShapeUtils::GetShape<ListShape>(rightShape))
			{
				auto cellShape = ShapeUtils::MergeShapes(listRight->GetElementShapes());
				Require(ShapeUtils::IsShape<VectorShape>(cellShape));

				auto vectorCell = ShapeUtils::GetShape<VectorShape>(cellShape);
				Require(CheckStaticEquality(vectorOffsets->GetSize(), vectorCell->GetSize()));
			}

			// Join call

			const auto [joinShapes, joinWriteShapes] = AnalyzeCall(joinFunction, {hashShapes.at(0), hashShapes.at(1), rightShape, countShapes.at(0), countShapes.at(1)}, {});
			Require(joinShapes.size() == 1);
			Require(ShapeUtils::IsShape<ListShape>(joinShapes.at(0)));

			const auto indexesShape = ShapeUtils::GetShape<ListShape>(joinShapes.at(0));
			const auto indexesSize = indexesShape->GetListSize();

			Require(ShapeUtils::IsSize<Shape::ConstantSize>(indexesSize));
			Require(ShapeUtils::GetSize<Shape::ConstantSize>(indexesSize)->GetValue() == 2);
			Require(CheckStaticTabular(indexesShape));

			// Return

			return {joinShapes, joinWriteShapes};
		}
		case BuiltinFunction::Primitive::GPUHashCreate:
		{
			// -- Vector input
			// Input: Vector<Size*>
			// Output: Vector<DynamicSize*>, Vector<DynamicSize*>
			//
			// -- List input
			// Input: List<k, {Vector<Size*>}>
			// Output: List<k, {Vector<DynamicSize*>}>, Vector<DynamicSize*>

			const auto dataShape = argumentShapes.at(0);
			if (const auto vectorShape = ShapeUtils::GetShape<VectorShape>(dataShape))
			{
				const auto writeShape = new VectorShape(new Shape::DynamicSize(m_call));

				if (const auto constantSize = ShapeUtils::GetSize<Shape::ConstantSize>(vectorShape->GetSize()))
				{
					const auto shift = Utils::Options::GetAlgorithm_HashSize();
					const auto powerSize = Utils::Math::Power2(constantSize->GetValue()) << shift;

					const auto valueShape = new VectorShape(new Shape::ConstantSize(powerSize));
					return {{valueShape, valueShape}, {writeShape, writeShape}};
				}

				const auto valueShape = new VectorShape(new Shape::DynamicSize(m_call));
				return {{valueShape, valueShape}, {writeShape, writeShape}};
			}
			else if (const auto listShape = ShapeUtils::GetShape<ListShape>(dataShape))
			{
				Require(CheckStaticTabular(listShape));

				auto mergedShape = ShapeUtils::MergeShapes(listShape->GetElementShapes());
				if (const auto vectorShape = ShapeUtils::GetShape<VectorShape>(mergedShape))
				{
					const auto writeShape1 = new VectorShape(new Shape::DynamicSize(m_call));
					const auto writeShape2 = new ListShape(listShape->GetListSize(), {writeShape1});

					if (const auto constantSize = ShapeUtils::GetSize<Shape::ConstantSize>(vectorShape->GetSize()))
					{
						const auto shift = Utils::Options::GetAlgorithm_HashSize();
						const auto powerSize = Utils::Math::Power2(constantSize->GetValue()) << shift;

						const auto valueShape = new VectorShape(new Shape::ConstantSize(powerSize));
						return {{new ListShape(listShape->GetListSize(), {valueShape}), valueShape}, {writeShape2, writeShape1}};
					}

					const auto valueShape = new VectorShape(new Shape::DynamicSize(m_call));
					return {{new ListShape(listShape->GetListSize(), {valueShape}), valueShape}, {writeShape2, writeShape1}};
				}
			}
			break;
		}
		case BuiltinFunction::Primitive::GPUHashJoinCount:
		{
			// -- Vector input
			// Input: Vector<Size1*>, Vector<Size2*>
			// Output: Vector<Size1*>, Vector<1>
			//
			// -- List input
			// Input: List<k, {Vector<Size1*>}>, List<k, {Vector<Size2*>}>
			// Output: Vector<Size1*>, Vector<1>

			auto keyShape = argumentShapes.at(0);
			auto rightShape = argumentShapes.at(1);

			Require(
				(ShapeUtils::IsShape<VectorShape>(keyShape) && ShapeUtils::IsShape<VectorShape>(rightShape)) ||
				(ShapeUtils::IsShape<ListShape>(keyShape) && ShapeUtils::IsShape<ListShape>(rightShape))
			);

			// Get the count shape (constant for runtime, compressed for codegen)

			const auto countShape = new VectorShape(new Shape::ConstantSize(1));
			const auto countWriteShape = new VectorShape(new Shape::CompressedSize(new DataObject(), new Shape::ConstantSize(1)));

			if (const auto vectorShape = ShapeUtils::GetShape<VectorShape>(rightShape))
			{
				// Ensure vector key

				Require(ShapeUtils::IsShape<VectorShape>(keyShape));

				// Return shapes

				std::vector<const Shape *> shapes({vectorShape, countShape});
				std::vector<const Shape *> writeShapes({vectorShape, countWriteShape});
				return {shapes, writeShapes};
			}
			else if (const auto listShape = ShapeUtils::GetShape<ListShape>(rightShape))
			{
				Require(ShapeUtils::IsShape<ListShape>(keyShape));

				// Return shapes

				auto cellShape = ShapeUtils::MergeShapes(listShape->GetElementShapes());
				Require(ShapeUtils::IsShape<VectorShape>(cellShape));

				std::vector<const Shape *> shapes({cellShape, countShape});
				std::vector<const Shape *> writeShapes({cellShape, countWriteShape});
				return {shapes, writeShapes};
			}
			break;
		}
		case BuiltinFunction::Primitive::GPUHashJoin:
		{
			// -- Unknown scalar constant
			// Intput: Vector<Size2*>, Vector<Size2*>, Vector<Size1*>, Vector<Size1*>, Vector<1>
			//       | List<k, {Vector<Size2*>}>, Vector<Size2*>, List<k, {Vector<Size1*>}>, Vector<Size1*>, Vector<1>
			// Output: List<2, {Vector<SizeDynamic>}>
			//
			// -- Known scalar constant
			// Intput: Vector<Size2*>, Vector<Size2*>, Vector<Size1*>, Vector<Size1*>, Vector<1> (value m)
			//       | List<k, {Vector<Size2*>}>, Vector<Size2*>, List<k, {Vector<Size1*>}>, Vector<Size1*>, Vector<1> (value m)
			// Output: List<2, {Vector<m>}>

			const auto offsetShape = argumentShapes.at(3);
			const auto countShape = argumentShapes.at(4);
			Require(ShapeUtils::IsShape<VectorShape>(offsetShape));
			Require(ShapeUtils::IsShape<VectorShape>(countShape));

			const auto vectorOffsets = ShapeUtils::GetShape<VectorShape>(offsetShape);
			const auto vectorCount = ShapeUtils::GetShape<VectorShape>(countShape);
			Require(CheckStaticScalar(vectorCount->GetSize()));

			// Hashtable

			const auto keyShape = argumentShapes.at(0);
			const auto valueShape = argumentShapes.at(1);

			Require(ShapeUtils::IsShape<VectorShape>(valueShape));
			const auto vectorValue = ShapeUtils::GetShape<VectorShape>(valueShape);

			// Left argument

			const auto rightShape = argumentShapes.at(2);
			if (const auto vectorRight = ShapeUtils::GetShape<VectorShape>(rightShape))
			{
				// Ensure equal size of key/value

				Require(ShapeUtils::IsShape<VectorShape>(keyShape));

				const auto vectorKey = ShapeUtils::GetShape<VectorShape>(keyShape);
				Require(CheckStaticEquality(vectorKey->GetSize(), vectorValue->GetSize()));

				// Offset shape equality

				Require(CheckStaticEquality(vectorOffsets->GetSize(), vectorRight->GetSize()));
			}
			else if (const auto listRight = ShapeUtils::GetShape<ListShape>(rightShape))
			{
				// Ensure equal size of key/value

				Require(ShapeUtils::IsShape<ListShape>(keyShape));

				const auto listKey = ShapeUtils::GetShape<ListShape>(keyShape);
				const auto cellKey = ShapeUtils::MergeShapes(listKey->GetElementShapes());
				Require(ShapeUtils::IsShape<VectorShape>(cellKey));

				const auto vectorCellKey = ShapeUtils::GetShape<VectorShape>(cellKey);
				Require(CheckStaticEquality(vectorCellKey->GetSize(), vectorValue->GetSize()));

				// Offset shape equality

				const auto cellShape = ShapeUtils::MergeShapes(listRight->GetElementShapes());
				Require(ShapeUtils::IsShape<VectorShape>(cellShape));

				const auto vectorCell = ShapeUtils::GetShape<VectorShape>(cellShape);
				Require(CheckStaticEquality(vectorOffsets->GetSize(), vectorCell->GetSize()));
			}

			if (arguments.size() > 0)
			{
				// Get the constant count parameter

				if (const auto [isConstant, value] = GetConstantArgument<std::int64_t>(arguments, arguments.size() - 1); isConstant)
				{
					Return(new ListShape(new Shape::ConstantSize(2), {new VectorShape(new Shape::ConstantSize(value))}));
				}
			}
			Return(new ListShape(new Shape::ConstantSize(2), {new VectorShape(new Shape::DynamicSize(m_call))}));
		}
		default:
		{
			Utils::Logger::LogError("Shape analysis is not supported for builtin function '" + function->GetName() + "'");
		}
	}

	ShapeError(function, argumentShapes);
}         

bool ShapeAnalysis::AnalyzeJoinArguments(const std::vector<const Shape *>& argumentShapes, const std::vector<const Operand *>& arguments)
{
#define RequireJoin(x) if (!(x)) return false

	// -- Vector join
	// Input: f, Vector<Size1*>, Vector<Size2*>
	// Output: List<2, {Vector<SizeDynamic>}>
	//
	// Require: f(Vector<Size1*>, Vector<Size2*>) == Vector
	//
	// -- List join
	// Input: f1, ..., fn, List<Size1*, {Vector<Size2*>}>, List<Size1*, {Vector<Size3*>}>
	// Output: List<2, {Vector<SizeDynamic>}>
	//
	// Require: f*(Vector<Size2*>, Vector<Size3*>) == Vector

	const auto functionCount = argumentShapes.size() - 2;
	const auto argumentShape1 = argumentShapes.at(functionCount);
	const auto argumentShape2 = argumentShapes.at(functionCount + 1);

	RequireJoin(
		(ShapeUtils::IsShape<VectorShape>(argumentShape1) && ShapeUtils::IsShape<VectorShape>(argumentShape2)) ||
		(ShapeUtils::IsShape<ListShape>(argumentShape1) && ShapeUtils::IsShape<ListShape>(argumentShape2))
	);

	if (const auto listShape1 = ShapeUtils::GetShape<ListShape>(argumentShape1))
	{
		const auto elementShapes1 = listShape1->GetElementShapes();
		const auto elementShapes2 = ShapeUtils::GetShape<ListShape>(argumentShape2)->GetElementShapes();

		auto elementCount1 = elementShapes1.size();
		auto elementCount2 = elementShapes2.size();
		if (elementCount1 == elementCount2)
		{
			RequireJoin(functionCount == 1 || elementCount1 == functionCount);
		}
		else if (elementCount1 == 1)
		{
			RequireJoin(functionCount == 1 || elementCount2 == functionCount);
		}
		else if (elementCount2 == 1)
		{
			RequireJoin(functionCount == 1 || elementCount1 == functionCount);
		}
		else
		{
			return false;
		}

		auto count = std::max({elementCount1, elementCount2, functionCount});
		for (auto i = 0u; i < count; ++i)
		{
			const auto type = arguments.at((functionCount == 1) ? 0 : i)->GetType();
			const auto function = TypeUtils::GetType<FunctionType>(type)->GetFunctionDeclaration();

			const auto l_inputShape1 = elementShapes1.at((elementCount1 == 1) ? 0 : i);
			const auto l_inputShape2 = new VectorShape(new Shape::ConstantSize(1));

			const auto [returnShapes, writeShapes] = AnalyzeCall(function, {l_inputShape1, l_inputShape2}, {});
			RequireJoin(ShapeUtils::IsSingleShape(returnShapes));
			RequireJoin(ShapeUtils::IsSingleShape(writeShapes));
			RequireJoin(ShapeUtils::IsShape<VectorShape>(ShapeUtils::GetSingleShape(returnShapes)));
			RequireJoin(ShapeUtils::IsShape<VectorShape>(ShapeUtils::GetSingleShape(writeShapes)));
		}
	}
	else
	{
		// If the inputs are vectors, require a single function

		RequireJoin(functionCount == 1);

		const auto type = arguments.at(0)->GetType();
		const auto function = TypeUtils::GetType<FunctionType>(type)->GetFunctionDeclaration();
		const auto l_inputShape2 = new VectorShape(new Shape::ConstantSize(1));

		const auto [returnShapes, writeShapes] = AnalyzeCall(function, {argumentShape1, l_inputShape2}, {});
		RequireJoin(ShapeUtils::IsSingleShape(returnShapes));
		RequireJoin(ShapeUtils::IsSingleShape(writeShapes));
		RequireJoin(ShapeUtils::IsShape<VectorShape>(ShapeUtils::GetSingleShape(returnShapes)));
		RequireJoin(ShapeUtils::IsShape<VectorShape>(ShapeUtils::GetSingleShape(writeShapes)));
	}
	return true;
}

[[noreturn]] void ShapeAnalysis::ShapeError(const FunctionDeclaration *function, const std::vector<const Shape *>& argumentShapes) const
{
	std::stringstream message;
	message << "Incompatible shapes [";

	bool first = true;
	for (const auto& argumentShape : argumentShapes)
	{
		if (!first)
		{
			message << ", ";
		}
		first = false;
		if (argumentShape)
		{
			message << *argumentShape;
		}
		else
		{
			message << "null";
		}
	}

	message << "] to function '" << function->GetName() << "'";
	Utils::Logger::LogError(message.str());
}

void ShapeAnalysis::Visit(const CastExpression *cast)
{
	// Traverse the expression

	cast->GetExpression()->Accept(*this);

	// Propagate the shape from the expression to the cast

	SetShapes(cast, GetShapes(cast->GetExpression()));
	SetWriteShapes(cast, GetWriteShapes(cast->GetExpression()));
}

void ShapeAnalysis::Visit(const Identifier *identifier)
{
	// Identifier shapes are propagated directly from the definition

	auto symbol = identifier->GetSymbol();

	SetShape(identifier, m_currentSet.first.at(symbol));
	SetWriteShape(identifier, m_currentSet.second.at(symbol));
}

void ShapeAnalysis::Visit(const VectorLiteral *literal)
{
	// Vector shapes are determined based on the number of static elements

	auto shape = new VectorShape(new Shape::ConstantSize(literal->GetCount()));

	SetShape(literal, shape);
	SetWriteShape(literal, shape);
}

const Shape *ShapeAnalysis::GetShape(const Operand *operand) const
{
	// Skip function operands

	if (TypeUtils::IsType<FunctionType>(operand->GetType()))
	{
		return nullptr;
	}

	// Utility for getting a single shape for an operand

	const auto& shapes = m_expressionShapes.at(operand);
	if (shapes.size() > 1)
	{
		Utils::Logger::LogError("Operand has more than one shape.");
	}
	return shapes.at(0);
}


const Shape *ShapeAnalysis::GetWriteShape(const Operand *operand) const
{
	// Skip function operands

	if (TypeUtils::IsType<FunctionType>(operand->GetType()))
	{
		return nullptr;
	}

	// Utility for getting a single shape for an operand

	const auto& shapes = m_writeShapes.at(operand);
	if (shapes.size() > 1)
	{
		Utils::Logger::LogError("Operand has more than one write shape.");
	}
	return shapes.at(0);
}

void ShapeAnalysis::SetShape(const Operand *operand, const Shape *shape)
{
	m_expressionShapes[operand] = {shape};
}

void ShapeAnalysis::SetWriteShape(const Operand *operand, const Shape *shape)
{
	m_writeShapes[operand] = {shape};
}

const std::vector<const Shape *>& ShapeAnalysis::GetShapes(const Expression *expression) const
{
	return m_expressionShapes.at(expression);
}

const std::vector<const Shape *>& ShapeAnalysis::GetWriteShapes(const Expression *expression) const
{
	return m_writeShapes.at(expression);
}

void ShapeAnalysis::SetShapes(const Expression *expression, const std::vector<const Shape *>& shapes)
{
	m_expressionShapes[expression] = shapes;
}

void ShapeAnalysis::SetWriteShapes(const Expression *expression, const std::vector<const Shape *>& shapes)
{
	m_writeShapes[expression] = shapes;
}

ShapeAnalysis::Properties ShapeAnalysis::InitialFlow() const
{
	// Add all global variables to the initial flow set

	Properties initialFlow;
	for (const auto module : m_program->GetModules())
	{
		for (const auto content : module->GetContents())
		{
			if (auto global = dynamic_cast<const GlobalDeclaration *>(content))
			{
				auto declaration = global->GetDeclaration();
				auto shape = ShapeUtils::ShapeFromType(declaration->GetType());

				initialFlow.first[declaration->GetSymbol()] = shape;
				initialFlow.second[declaration->GetSymbol()] = shape;
			}
		}
	}
	return initialFlow;
}

ShapeAnalysis::Properties ShapeAnalysis::Merge(const Properties& s1, const Properties& s2) const
{
	// Merge the maps using a shape merge operation on each element

	Properties outSet(s1);
	MergeShapes(outSet.first, s2.first);
	MergeShapes(outSet.second, s2.second);
	return outSet;
}

void ShapeAnalysis::MergeShapes(::Analysis::Map<SymbolObject, ShapeAnalysisValue>& outMap, const ::Analysis::Map<SymbolObject, ShapeAnalysisValue>& otherMap) const
{
	for (const auto& [symbol, shape] : otherMap)
	{
		auto it = outMap.find(symbol);
		if (it != outMap.end())
		{
			// Merge shapes according to the rules

			outMap[symbol] = ShapeUtils::MergeShape(shape, it->second);
		}
		else
		{
			outMap.insert({symbol, shape});
		}
	}
}

}
}
