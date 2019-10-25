#include "Analysis/Shape/ShapeAnalysisHelper.h"

#include <algorithm>

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

	SetShapes(call, AnalyzeCall(call->GetFunctionLiteral()->GetFunction(), argumentShapes, arguments));
	m_call = nullptr;
}

bool ShapeAnalysisHelper::CheckStaticScalar(const Shape::Size *size, bool enforce) const
{
	// Check if the size is a static scalar, enforcing if required

	if (ShapeUtils::IsSize<Shape::ConstantSize>(size))
	{
		return ShapeUtils::IsScalarSize(size);
	}
	return !enforce;
}

bool ShapeAnalysisHelper::CheckStaticEquality(const Shape::Size *size1, const Shape::Size *size2, bool enforce) const
{
	if (*size1 == *size2)
	{
		return true;
	}
	return !enforce;
}

bool ShapeAnalysisHelper::CheckStaticTabular(const ListShape *listShape, bool enforce) const
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

			if (!CheckStaticEquality(vectorShape1->GetSize(), vectorShape2->GetSize(), enforce))
			{
				return false;
			}
		}
	}

	return true;
}

bool ShapeAnalysisHelper::HasConstantArgument(const std::vector<HorseIR::Operand *>& arguments, unsigned int index) const
{
	if (arguments.size() <= index)
	{
		return false;
	}
	return ValueAnalysisHelper::IsConstant(arguments.at(index));
}

std::vector<const Shape *> ShapeAnalysisHelper::AnalyzeCall(const HorseIR::FunctionDeclaration *function, const std::vector<const Shape *>& argumentShapes, const std::vector<HorseIR::Operand *>& arguments)
{
	switch (function->GetKind())
	{
		case HorseIR::FunctionDeclaration::Kind::Builtin:
			return AnalyzeCall(static_cast<const HorseIR::BuiltinFunction *>(function), argumentShapes, arguments);
		case HorseIR::FunctionDeclaration::Kind::Definition:
			return AnalyzeCall(static_cast<const HorseIR::Function *>(function), argumentShapes, arguments);
		default:
			Utils::Logger::LogError("Unsupported function kind");
	}
}

std::vector<const Shape *> ShapeAnalysisHelper::AnalyzeCall(const HorseIR::Function *function, const std::vector<const Shape *>& argumentShapes, const std::vector<HorseIR::Operand *>& arguments)
{
	// Accumulate default return shapes based on the argument types

	const auto& returnTypes = function->GetReturnTypes();
	if (returnTypes.size() == 1)
	{
		return {ShapeUtils::ShapeFromType(returnTypes.at(0), m_call)};
	}
	else
	{
		std::vector<const Shape *> returnShapes;
		auto tag = 1u;
		for (const auto returnType : function->GetReturnTypes())
		{
			returnShapes.push_back(ShapeUtils::ShapeFromType(returnType, m_call, tag++));
		}
		return returnShapes;
	}
}

std::vector<const Shape *> ShapeAnalysisHelper::AnalyzeCall(const HorseIR::BuiltinFunction *function, const std::vector<const Shape *>& argumentShapes, const std::vector<HorseIR::Operand *>& arguments)
{
	switch (function->GetPrimitive())
	{
#define Require(x) if (!(x)) break

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
			// -- Propagate size
			// Input: Vector<Size1*>
			// Output: Vector<Size1*>

			const auto argumentShape = argumentShapes.at(0);
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

			if (*argumentSize1 == *argumentSize2)
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
			else if (ShapeUtils::IsSize<Shape::ConstantSize>(argumentSize1) && ShapeUtils::IsSize<Shape::ConstantSize>(argumentSize1))
			{
				// Error case, unequal constants where neither is a scalar

				auto constant1 = ShapeUtils::GetSize<Shape::ConstantSize>(argumentSize1)->GetValue();
				auto constant2 = ShapeUtils::GetSize<Shape::ConstantSize>(argumentSize2)->GetValue();
				Utils::Logger::LogError("Binary function '" + function->GetName() + "' requires vector of same length (or broadcast) [" + std::to_string(constant1) + " != " + std::to_string(constant2) + "]");
			}
			else
			{
				// Determine at runtime

				size = new Shape::DynamicSize(m_call);
			}
			return {new VectorShape(size)};
		}

		// Algebraic Unary
		case HorseIR::BuiltinFunction::Primitive::Unique:
		{
			// -- Output is always dynamic
			// Input: Vector<Size*>
			// Output: Vector<SizeDynamic>

			const auto argumentShape = argumentShapes.at(0);
			Require(ShapeUtils::IsShape<VectorShape>(argumentShape));
			return {new VectorShape(new Shape::DynamicSize(m_call))};
		}
		case HorseIR::BuiltinFunction::Primitive::Range:
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

			if (HasConstantArgument(arguments, 0))
			{
				auto value = ValueAnalysisHelper::GetScalar<std::int64_t>(arguments.at(0));
				return {new VectorShape(new Shape::ConstantSize(value))};
			}
			return {new VectorShape(new Shape::DynamicSize(m_call))};
		}
		case HorseIR::BuiltinFunction::Primitive::Factorial:
		case HorseIR::BuiltinFunction::Primitive::Reverse:
		{
			// -- Propagate size
			// Input: Vector<Size*>
			// Output: Vector<Size*>

			const auto argumentShape = argumentShapes.at(0);
			Require(ShapeUtils::IsShape<VectorShape>(argumentShape));
			return {argumentShape};
		}
		case HorseIR::BuiltinFunction::Primitive::Random:
		case HorseIR::BuiltinFunction::Primitive::Seed:
		{
			// -- Scalar
			// Input: Vector<1>
			// Output: Vector<1>

			const auto argumentShape = argumentShapes.at(0);
			Require(ShapeUtils::IsShape<VectorShape>(argumentShape));

			const auto vectorShape = ShapeUtils::GetShape<VectorShape>(argumentShape);
			Require(CheckStaticScalar(vectorShape->GetSize()));

			return {new VectorShape(new Shape::ConstantSize(1))};
		}
		case HorseIR::BuiltinFunction::Primitive::Flip:
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

			return {new ListShape(listShape->GetListSize(), elementShapes)};
		}
		case HorseIR::BuiltinFunction::Primitive::Where:
		{
			// -- Compress itself(!) by the mask
			// Input: Vector<Size*> (mask)
			// Output: Vector<Size*[mask]>

			const auto argumentShape = argumentShapes.at(0);
			Require(ShapeUtils::IsShape<VectorShape>(argumentShape));

			const auto argumentSize = ShapeUtils::GetShape<VectorShape>(argumentShape)->GetSize();
			return {new VectorShape(new Shape::CompressedSize(arguments.at(0), argumentSize))};
		}
		case HorseIR::BuiltinFunction::Primitive::Group:
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
			return {new DictionaryShape(
				new VectorShape(new Shape::DynamicSize(m_call, 1)),
				new VectorShape(new Shape::DynamicSize(m_call, 2))
			)};
		}

		// Algebraic Binary
		case HorseIR::BuiltinFunction::Primitive::Append:
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
					auto length1 = ShapeUtils::IsSize<Shape::ConstantSize>(vectorSize1);
					auto length2 = ShapeUtils::IsSize<Shape::ConstantSize>(vectorSize2);
					return {new VectorShape(new Shape::ConstantSize(length1 + length2))};
				}
				return {new VectorShape(new Shape::DynamicSize(m_call))};
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
					return {new ListShape(new Shape::ConstantSize(length1 + length2), elementShapes)};
				}

				auto mergedShapes1 = ShapeUtils::MergeShapes(elementShapes1);
				auto mergedShapes2 = ShapeUtils::MergeShapes(elementShapes2);
				auto mergedShape = ShapeUtils::MergeShape(mergedShapes1, mergedShapes2);
				return {new ListShape(new Shape::DynamicSize(m_call), {mergedShape})};
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
						return {new EnumerationShape(keyShape, new VectorShape(new Shape::ConstantSize(length1 + length2)))};
					}
					return {new EnumerationShape(keyShape, new VectorShape(new Shape::DynamicSize(m_call)))};
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
						return {new EnumerationShape(keyShape, new ListShape(listValueShape->GetListSize(), newElementShapes))};
					}
					return {new EnumerationShape(keyShape, new ListShape(listValueShape->GetListSize(), {new VectorShape(new Shape::DynamicSize(m_call))}))};
				}
			}
			break;
		}
		case HorseIR::BuiltinFunction::Primitive::Replicate:
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

			if (HasConstantArgument(arguments, 0))
			{
				auto value = ValueAnalysisHelper::GetScalar<std::int64_t>(arguments.at(0));
				if (ShapeUtils::IsShape<VectorShape>(argumentShape2))
				{
					return {new ListShape(new Shape::ConstantSize(value), {argumentShape2})};
				}
				else if (const auto listShape = ShapeUtils::GetShape<ListShape>(argumentShape2))
				{
					if (const auto constantSize = ShapeUtils::GetSize<Shape::ConstantSize>(listShape->GetListSize()))
					{
						auto listLength = constantSize->GetValue();

						const auto& elementShapes = listShape->GetElementShapes();
						if (elementShapes.size() == 1)
						{
							return {new ListShape(new Shape::ConstantSize(value * listLength), {elementShapes.at(0)})};
						}

						std::vector<const Shape *> newElementShapes;
						for (auto i = 0u; i < listLength; ++i)
						{
							newElementShapes.insert(std::end(newElementShapes), std::begin(elementShapes), std::end(elementShapes));
						}
						return {new ListShape(new Shape::ConstantSize(value * listLength), newElementShapes)};
					}
					return {new ListShape(new Shape::DynamicSize(m_call), {ShapeUtils::MergeShapes(listShape->GetElementShapes())})};
				}
			}
			else
			{
				if (ShapeUtils::IsShape<VectorShape>(argumentShape2))
				{
					return {new ListShape(new Shape::DynamicSize(m_call), {argumentShape2})};
				}
				else if (const auto listShape = ShapeUtils::GetShape<ListShape>(argumentShape2))
				{
					const auto elementShape = ShapeUtils::MergeShapes(listShape->GetElementShapes());
					return {new ListShape(new Shape::DynamicSize(m_call), {elementShape})};
				}
			}
			break;
		}
		case HorseIR::BuiltinFunction::Primitive::Like:
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
			return {argumentShape1};
		}
		case HorseIR::BuiltinFunction::Primitive::Compress:
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
			return {new VectorShape(new Shape::CompressedSize(arguments.at(0), argumentSize2))};
		}
		case HorseIR::BuiltinFunction::Primitive::Random_k:
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

			if (HasConstantArgument(arguments, 0))
			{
				auto value = ValueAnalysisHelper::GetScalar<std::int64_t>(arguments.at(0));
				return {new VectorShape(new Shape::ConstantSize(value))};
			}
			return {new VectorShape(new Shape::DynamicSize(m_call))};
		}
		case HorseIR::BuiltinFunction::Primitive::IndexOf:
		{
			// -- Propagate right
			// Input: Vector<Size1*>, Vector<Size2*>
			// Output: Vector<Size2*>

			const auto argumentShape1 = argumentShapes.at(0);
			const auto argumentShape2 = argumentShapes.at(1);
			Require(ShapeUtils::IsShape<VectorShape>(argumentShape1));
			Require(ShapeUtils::IsShape<VectorShape>(argumentShape2));
			return {argumentShape1};
		}
		case HorseIR::BuiltinFunction::Primitive::Take:
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
			Require(CheckStaticScalar(vectorShape1->GetSize()));

			if (HasConstantArgument(arguments, 0))
			{
				auto value = ValueAnalysisHelper::GetScalar<std::int64_t>(arguments.at(0));
				value = (value < 0) ? -value : value;
				return {new VectorShape(new Shape::ConstantSize(value))};
			}
			return {new VectorShape(new Shape::DynamicSize(m_call))};
		}
		case HorseIR::BuiltinFunction::Primitive::Drop:
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
			Require(CheckStaticScalar(vectorShape1->GetSize()));

			const auto vectorSize2 = ShapeUtils::GetShape<VectorShape>(argumentShape2)->GetSize();
			if (HasConstantArgument(arguments, 0) && ShapeUtils::IsSize<Shape::ConstantSize>(vectorSize2))
			{
				// Compute the modification length

				auto modLength = ValueAnalysisHelper::GetScalar<std::int64_t>(arguments.at(0));
				modLength = (modLength < 0) ? -modLength : modLength;

				// Compute the new vector length, ceil at 0

				auto vectorLength = ShapeUtils::GetSize<Shape::ConstantSize>(vectorSize2)->GetValue();
				auto value = vectorLength - modLength;
				value = (value < 0) ? 0 : value;

				return {new VectorShape(new Shape::ConstantSize(value))};
			}
			return {new VectorShape(new Shape::DynamicSize(m_call))};
		}
		case HorseIR::BuiltinFunction::Primitive::Order:
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
				return {argumentShape1};
			}
			else if (const auto listShape = ShapeUtils::GetShape<ListShape>(argumentShape1))
			{
				Require(CheckStaticEquality(listShape->GetListSize(), vectorShape2->GetSize()));
				Require(CheckStaticTabular(listShape));
				return {ShapeUtils::MergeShapes(listShape->GetElementShapes())};
			}
			break;
		}
		case HorseIR::BuiltinFunction::Primitive::Member:
		{
			// -- Propagate left shape
			// Input: Vector<Size1*>, Vector<Size2*>
			// Output: Vector<Size1*>

			const auto argumentShape1 = argumentShapes.at(0);
			const auto argumentShape2 = argumentShapes.at(1);
			Require(ShapeUtils::IsShape<VectorShape>(argumentShape1));
			Require(ShapeUtils::IsShape<VectorShape>(argumentShape2));
			return {argumentShape1};
		}
		case HorseIR::BuiltinFunction::Primitive::Vector:
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

			if (HasConstantArgument(arguments, 0))
			{
				auto value = ValueAnalysisHelper::GetScalar<std::int64_t>(arguments.at(0));
				return {new VectorShape(new Shape::ConstantSize(value))};
			}
			return {new VectorShape(new Shape::DynamicSize(m_call))};
		}

		// Reduction
		case HorseIR::BuiltinFunction::Primitive::Length:
		case HorseIR::BuiltinFunction::Primitive::Sum:
		case HorseIR::BuiltinFunction::Primitive::Average:
		case HorseIR::BuiltinFunction::Primitive::Minimum:
		case HorseIR::BuiltinFunction::Primitive::Maximum:
		{
			// -- Reduce to a single value
			// Input: Vector<Size*>
			// Output: Vector<1>

			const auto argumentShape = argumentShapes.at(0);
			Require(ShapeUtils::IsShape<VectorShape>(argumentShape));
			return {new VectorShape(new Shape::ConstantSize(1))};
		}

		// List
		case HorseIR::BuiltinFunction::Primitive::Raze:
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
						return {new VectorShape(new Shape::ConstantSize(listSize->GetValue() * vectorSize->GetValue()))};
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
							return {new VectorShape(new Shape::DynamicSize(m_call))};
						}
					}
					return {new VectorShape(new Shape::ConstantSize(count))};
				}
			}
			return {new VectorShape(new Shape::DynamicSize(m_call))};
		}
		case HorseIR::BuiltinFunction::Primitive::List:
		{
			// -- Static elements
			// Input: Shape1*, ..., ShapeN*
			// Output: List<N, {Shape1*, ..., ShapeN*}>

			return {new ListShape(new Shape::ConstantSize(arguments.size()), argumentShapes)};
		}
		case HorseIR::BuiltinFunction::Primitive::ToList:
		{
			// -- Expand vector to list
			// Input: Vector<Size*>
			// Output: List<Size*, {Vector<1>}>

			const auto argumentShape = argumentShapes.at(0);
			Require(ShapeUtils::IsShape<VectorShape>(argumentShape));

			const auto vectorShape = ShapeUtils::GetShape<VectorShape>(argumentShape);
			return {new ListShape(vectorShape->GetSize(), {new VectorShape(new Shape::ConstantSize(1))})};
		}
		case HorseIR::BuiltinFunction::Primitive::Each:
		{
			// -- Function call
			// Input: f, List<Size*, {Shape*}>
			// Output: List<Size*, f(Shape*)>

			const auto type = arguments.at(0)->GetType();
			const auto function = HorseIR::TypeUtils::GetType<HorseIR::FunctionType>(type)->GetFunctionDeclaration();

			const auto argumentShape = argumentShapes.at(1);
			Require(ShapeUtils::IsShape<ListShape>(argumentShape));

			const auto listShape = ShapeUtils::GetShape<ListShape>(argumentShape);

			std::vector<const Shape *> newElementShapes;
			for (const auto& elementShape : listShape->GetElementShapes())
			{
				const auto shapes = AnalyzeCall(function, {elementShape}, {});
				Require(ShapeUtils::IsSingleShape(shapes));
				newElementShapes.push_back(ShapeUtils::GetSingleShape(shapes));
			}
			return {new ListShape(listShape->GetListSize(), newElementShapes)};
		}
		case HorseIR::BuiltinFunction::Primitive::EachItem:
		{
			// -- Function call
			// Input: f, List<Size*, {Shape1*}>, List<Size*, {Shape2*}>
			// Output: List<Size*, f(Shape1*, Shape2*)>

			const auto type = arguments.at(0)->GetType();
			const auto function = HorseIR::TypeUtils::GetType<HorseIR::FunctionType>(type)->GetFunctionDeclaration();

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
			for (auto i = 0u; i < count; ++i)
			{
				const auto l_inputShape1 = elementShapes1.at((elementCount1 == 1) ? 0 : i);
				const auto l_inputShape2 = elementShapes2.at((elementCount2 == 1) ? 0 : i);

				const auto shapes = AnalyzeCall(function, {l_inputShape1, l_inputShape2}, {});
				Require(ShapeUtils::IsSingleShape(shapes));
				newElementShapes.push_back(ShapeUtils::GetSingleShape(shapes));
			}
			return {new ListShape(listShape1->GetListSize(), newElementShapes)};
		}
		case HorseIR::BuiltinFunction::Primitive::EachLeft:
		{
			// -- Function call
			// Input: f, List<Size*, {Shape1*}>, Shape2*
			// Output: List<Size*, {f(Shape1*, Shape2*)}>

			const auto type = arguments.at(0)->GetType();
			const auto function = HorseIR::TypeUtils::GetType<HorseIR::FunctionType>(type)->GetFunctionDeclaration();

			const auto argumentShape1 = argumentShapes.at(1);
			const auto argumentShape2 = argumentShapes.at(2);
			Require(ShapeUtils::IsShape<ListShape>(argumentShape1));

			const auto listShape1 = ShapeUtils::GetShape<ListShape>(argumentShape1);

			std::vector<const Shape *> newElementShapes;
			for (const auto& elementShape1 : listShape1->GetElementShapes())
			{
				const auto shapes = AnalyzeCall(function, {elementShape1, argumentShape2}, {});
				Require(ShapeUtils::IsSingleShape(shapes));
				newElementShapes.push_back(ShapeUtils::GetSingleShape(shapes));
			}
			return {new ListShape(listShape1->GetListSize(), newElementShapes)};
		}
		case HorseIR::BuiltinFunction::Primitive::EachRight:
		{
			// -- Function call
			// Input: f, Shape1*, List<Size*, {Shape2*}>
			// Output: List<Size*, {f(Shape1*, Shape2*)}>

			const auto type = arguments.at(0)->GetType();
			const auto function = HorseIR::TypeUtils::GetType<HorseIR::FunctionType>(type)->GetFunctionDeclaration();

			const auto argumentShape1 = argumentShapes.at(1);
			const auto argumentShape2 = argumentShapes.at(2);
			Require(ShapeUtils::IsShape<ListShape>(argumentShape2));

			const auto listShape2 = ShapeUtils::GetShape<ListShape>(argumentShape2);

			std::vector<const Shape *> newElementShapes;
			for (const auto& elementShape2 : listShape2->GetElementShapes())
			{
				const auto shapes = AnalyzeCall(function, {argumentShape1, elementShape2}, {});
				Require(ShapeUtils::IsSingleShape(shapes));
				newElementShapes.push_back(ShapeUtils::GetSingleShape(shapes));
			}
			return {new ListShape(listShape2->GetListSize(), newElementShapes)};
		}
		case HorseIR::BuiltinFunction::Primitive::Match:
		{
			// -- Any to scalar
			// Input: Shape1*, Shape2*
			// Output: Vector<1>

			return {new VectorShape(new Shape::ConstantSize(1))};
		}

		// Database
		case HorseIR::BuiltinFunction::Primitive::Enum:
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
			return {new EnumerationShape(argumentShape1, argumentShape2)};
		}
		case HorseIR::BuiltinFunction::Primitive::Dictionary:
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
			return {new DictionaryShape(argumentShape1, argumentShape2)};
		}
		case HorseIR::BuiltinFunction::Primitive::Table:
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
			return {new TableShape(vectorSize1, rowVector->GetSize())};
		}
		case HorseIR::BuiltinFunction::Primitive::KeyedTable:
		{
			// -- Create compound type
			// Input: TableShape1, TableShape2
			// Output: KTable<TableShape1, TableShape2>

			const auto argumentShape1 = argumentShapes.at(0);
			const auto argumentShape2 = argumentShapes.at(1);
			Require(ShapeUtils::IsShape<TableShape>(argumentShape1));
			Require(ShapeUtils::IsShape<TableShape>(argumentShape2));
			return {new KeyedTableShape(ShapeUtils::GetShape<TableShape>(argumentShape1), ShapeUtils::GetShape<TableShape>(argumentShape2))};
		}
		case HorseIR::BuiltinFunction::Primitive::Keys:
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
					return {keyShape};
				}
				return {new ListShape(new Shape::DynamicSize(m_call), {keyShape})};
			}
			else if (const auto tableShape = ShapeUtils::GetShape<TableShape>(argumentShape))
			{
				return {new VectorShape(tableShape->GetColumnsSize())};
			}
			else if (const auto kTableShape = ShapeUtils::GetShape<KeyedTableShape>(argumentShape))
			{
				return {kTableShape->GetKeyShape()};
			}
			else if (const auto enumShape = ShapeUtils::GetShape<EnumerationShape>(argumentShape))
			{
				return {enumShape->GetKeyShape()};
			}
			break;
		}
		case HorseIR::BuiltinFunction::Primitive::Values:
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
			// Output: Shape2*

			const auto argumentShape = argumentShapes.at(0);
			if (const auto dictionaryShape = ShapeUtils::GetShape<DictionaryShape>(argumentShape))
			{
				const auto valueShape = dictionaryShape->GetValueShape();
				if (ShapeUtils::IsShape<ListShape>(valueShape))
				{
					return {valueShape};
				}
				return {new ListShape(new Shape::DynamicSize(m_call), {valueShape})};
			}
			else if (const auto tableShape = ShapeUtils::GetShape<TableShape>(argumentShape))
			{
				const auto colsSize = tableShape->GetColumnsSize();
				const auto rowsSize = tableShape->GetRowsSize();
				return {new ListShape(colsSize, {new VectorShape(rowsSize)})};
			}
			else if (const auto kTableShape = ShapeUtils::GetShape<KeyedTableShape>(argumentShape))
			{
				return {kTableShape->GetValueShape()};
			}
			else if (const auto enumShape = ShapeUtils::GetShape<EnumerationShape>(argumentShape))
			{
				return {enumShape->GetValueShape()};
			}
			break;
		}
		case HorseIR::BuiltinFunction::Primitive::Meta:
		{
			// -- Always dynamic
			// Input: Table* || KeyedTable*
			// Output: Table<SizeDynamic1, SizeDynamic2>

			const auto argumentShape = argumentShapes.at(0);
			Require(ShapeUtils::IsShape<TableShape>(argumentShape) || ShapeUtils::IsShape<KeyedTableShape>(argumentShape));
			return {new TableShape(new Shape::DynamicSize(m_call, 1), new Shape::DynamicSize(m_call, 2))};
		}
		case HorseIR::BuiltinFunction::Primitive::Fetch:
		{
			// -- Enum unpacking (get the value)
			// Input: Enum<Shape1*, Shape2*>
			// Output: Shape2*

			const auto argumentShape = argumentShapes.at(0);
			Require(ShapeUtils::IsShape<EnumerationShape>(argumentShape));
			return {ShapeUtils::GetShape<EnumerationShape>(argumentShape)->GetValueShape()};
		}
		case HorseIR::BuiltinFunction::Primitive::ColumnValue:
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
			return {new VectorShape(tableShape->GetRowsSize())};
		}
		case HorseIR::BuiltinFunction::Primitive::LoadTable:
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

			if (HasConstantArgument(arguments, 0))
			{
				auto tableName = ValueAnalysisHelper::GetScalar<const HorseIR::SymbolValue *>(arguments.at(0))->GetName();
				return {new TableShape(new Shape::SymbolSize("data." + tableName + ".cols"), new Shape::SymbolSize("data." + tableName + ".rows"))};
			}
			return {new TableShape(new Shape::DynamicSize(m_call, 1), new Shape::DynamicSize(m_call, 2))};
		}
		case HorseIR::BuiltinFunction::Primitive::JoinIndex:
		{
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

			Require(
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
					Require(functionCount == 1 || elementCount1 == functionCount);
				}
				else if (elementCount1 == 1)
				{
					Require(functionCount == 1 || elementCount2 == functionCount);
				}
				else if (elementCount2 == 1)
				{
					Require(functionCount == 1 || elementCount1 == functionCount);
				}
				else
				{
					Require(false);
				}

				auto count = std::max({elementCount1, elementCount2, functionCount});
				for (auto i = 0u; i < count; ++i)
				{
					const auto type = arguments.at((functionCount == 1) ? 0 : i)->GetType();
					const auto function = HorseIR::TypeUtils::GetType<HorseIR::FunctionType>(type)->GetFunctionDeclaration();

					const auto l_inputShape1 = elementShapes1.at((elementCount1 == 1) ? 0 : i);
					const auto l_inputShape2 = elementShapes2.at((elementCount2 == 1) ? 0 : i);

					const auto returnShapes = AnalyzeCall(function, {l_inputShape1, l_inputShape2}, {});
					Require(ShapeUtils::IsSingleShape(returnShapes));
					Require(ShapeUtils::IsShape<VectorShape>(ShapeUtils::GetSingleShape(returnShapes)));
				}
			}
			else
			{
				// If the inputs are vectors, require a single function

				Require(functionCount == 1);

				const auto type = arguments.at(0)->GetType();
				const auto function = HorseIR::TypeUtils::GetType<HorseIR::FunctionType>(type)->GetFunctionDeclaration();

				const auto returnShapes = AnalyzeCall(function, {argumentShape1, argumentShape2}, {});
				Require(ShapeUtils::IsSingleShape(returnShapes));
				Require(ShapeUtils::IsShape<VectorShape>(ShapeUtils::GetSingleShape(returnShapes)));
			}
			return {new ListShape(new Shape::ConstantSize(2), {new VectorShape(new Shape::DynamicSize(m_call))})};
		}

		// Indexing
		case HorseIR::BuiltinFunction::Primitive::Index:
		{
			// -- Right propagation
			// Input: Vector<Size1*>, Vector<Size2*>
			// Output: Vector<Size2*>

			const auto argumentShape1 = argumentShapes.at(0);
			const auto argumentShape2 = argumentShapes.at(1);
			Require(ShapeUtils::IsShape<VectorShape>(argumentShape1));
			Require(ShapeUtils::IsShape<VectorShape>(argumentShape2));
			return {argumentShape2};
		}
		case HorseIR::BuiltinFunction::Primitive::IndexAssignment:
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
			return {argumentShape1};
		}

		// Other
		case HorseIR::BuiltinFunction::Primitive::LoadCSV:
		{
			// -- Dynamic table load
			// Input: Vector<1>
			// Output: Table<SizeDynamic1, SizeDynamic2>

			const auto argumentShape = argumentShapes.at(0);
			Require(ShapeUtils::IsShape<VectorShape>(argumentShape));

			const auto vectorShape = ShapeUtils::GetShape<VectorShape>(argumentShape);
			Require(CheckStaticScalar(vectorShape->GetSize()));
			return {new TableShape(new Shape::DynamicSize(m_call, 1), new Shape::DynamicSize(m_call, 2))};
		}
		case HorseIR::BuiltinFunction::Primitive::Print:
		case HorseIR::BuiltinFunction::Primitive::Format:
		case HorseIR::BuiltinFunction::Primitive::String:
		{
			// -- Any shape
			// Input: Shape*
			// Outout: Vector<1>

			return {new VectorShape(new Shape::ConstantSize(1))};
		}
		case HorseIR::BuiltinFunction::Primitive::SubString:
		{
			// -- Substring of vector
			// Input: Vector<Size*>, Vector<1>, Vector<1>
			// Output: Vector<Size*>

			const auto argumentShape1 = argumentShapes.at(0);
			const auto argumentShape2 = argumentShapes.at(1);
			const auto argumentShape3 = argumentShapes.at(2);
			Require(ShapeUtils::IsShape<VectorShape>(argumentShape1));
			Require(ShapeUtils::IsShape<VectorShape>(argumentShape2));
			Require(ShapeUtils::IsShape<VectorShape>(argumentShape3));

			const auto vectorShape2 = ShapeUtils::GetShape<VectorShape>(argumentShape2);
			const auto vectorShape3 = ShapeUtils::GetShape<VectorShape>(argumentShape3);
			Require(CheckStaticScalar(vectorShape2->GetSize()));
			Require(CheckStaticScalar(vectorShape3->GetSize()));
			return {argumentShape1};
		}
		default:
		{
			Utils::Logger::LogError("Shape analysis is not supported for builtin function '" + function->GetName() + "'");
		}
	}

	ShapeError(function, argumentShapes);
}         

[[noreturn]] void ShapeAnalysisHelper::ShapeError(const HorseIR::FunctionDeclaration *function, const std::vector<const Shape *>& argumentShapes) const
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
		message << *argumentShape;
	}

	message << "] to function '" << function->GetName() << "'";
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
	// Identifier shapes are propagated directly from the definition

	SetShape(identifier, m_properties.at(identifier->GetSymbol()));
}

void ShapeAnalysisHelper::Visit(const HorseIR::VectorLiteral *literal)
{
	// Vector shapes are determined based on the number of static elements

	SetShape(literal, new VectorShape(new Shape::ConstantSize(literal->GetCount())));
}

const Shape *ShapeAnalysisHelper::GetShape(const HorseIR::Operand *operand) const
{
	// Skip function operands

	if (HorseIR::TypeUtils::IsType<HorseIR::FunctionType>(operand->GetType()))
	{
		return nullptr;
	}

	// Utility for getting a single shape for an operand

	const auto& shapes = m_shapes.at(operand);
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
