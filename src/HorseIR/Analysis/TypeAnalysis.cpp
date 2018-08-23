#include "HorseIR/Analysis/TypeAnalysis.h"

#include "HorseIR/TypeUtils.h"
#include "HorseIR/Tree/BuiltinMethod.h"
#include "HorseIR/Tree/Method.h"
#include "HorseIR/Tree/MethodDeclaration.h"
#include "HorseIR/Tree/Program.h"
#include "HorseIR/Tree/Expressions/CallExpression.h"
#include "HorseIR/Tree/Expressions/CastExpression.h"
#include "HorseIR/Tree/Expressions/Expression.h"
#include "HorseIR/Tree/Expressions/Identifier.h"
#include "HorseIR/Tree/Expressions/Literals/FunctionLiteral.h"
#include "HorseIR/Tree/Statements/AssignStatement.h"
#include "HorseIR/Tree/Statements/ReturnStatement.h"
#include "HorseIR/Tree/Types/DictionaryType.h"
#include "HorseIR/Tree/Types/ListType.h"
#include "HorseIR/Tree/Types/FunctionType.h"
#include "HorseIR/Tree/Types/TableType.h"

#include "Utils/Logger.h"

namespace HorseIR {

void TypeAnalysis::Analyze(Program *program)
{
	program->Accept(*this);
}

void TypeAnalysis::Visit(Method *method)
{
	m_currentMethod = method;
	ForwardTraversal::Visit(method);
	m_currentMethod = nullptr;
}

void TypeAnalysis::Visit(AssignStatement *assign)
{
	// Traverse all children of the assignment

	ForwardTraversal::Visit(assign);

	// Update the declaration with the shape propagated from the expression

	auto declarationType = assign->GetDeclaration()->GetType();
	auto expressionType = assign->GetExpression()->GetType();

	if (*declarationType != *expressionType)
	{
		Utils::Logger::LogError("Expression type '" + expressionType->ToString() + "' does not match destination type '" + declarationType->ToString() + "'");
	}
}

void TypeAnalysis::Visit(CallExpression *call)
{
	// Collect type information for the arguments

	ForwardTraversal::Visit(call);

	// Analyze the function according to the type rules

	auto method = call->GetMethod();
	std::vector<Type *> argumentTypes;
	for (auto& argument : call->GetArguments())
	{
		argumentTypes.push_back(argument->GetType());
	}
	call->SetType(AnalyzeCall(method, argumentTypes));
}

Type *TypeAnalysis::AnalyzeCall(const MethodDeclaration *method, const std::vector<Type *>& argumentTypes)
{
	switch (method->GetKind())
	{
		case MethodDeclaration::Kind::Builtin:
			return AnalyzeCall(static_cast<const BuiltinMethod *>(method), argumentTypes);
		case MethodDeclaration::Kind::Definition:
			return AnalyzeCall(static_cast<const Method *>(method), argumentTypes);
		default:
			Utils::Logger::LogError("Unsupported method kind");
	}
}

[[noreturn]] void TypeError(const MethodDeclaration *method, const std::vector<Type *>& argumentTypes)
{
	std::string message = "Incompatible arguments [";
	bool first = true;
	for (const auto& argumentType : argumentTypes)
	{
		if (!first)
		{
			message += ", ";
		}
		first = false;
		message += argumentType->ToString();
	}
	message += "] to function '" + method->GetName() + "'";
	Utils::Logger::LogError(message);
}

Type *TypeAnalysis::AnalyzeCall(const Method *method, const std::vector<Type *>& argumentTypes)
{
	auto argumentCount = argumentTypes.size();
	auto parameterCount = method->GetParameterCount();
	if (argumentCount != parameterCount)
	{
		TypeError(method, argumentTypes);
	}

	unsigned int index = 0;
	for (const auto& parameter : method->GetParameters())
	{
		const auto& argumentType = argumentTypes.at(index);
		if (*parameter->GetType() != *argumentType)
		{
			TypeError(method, argumentTypes);
		}
		index++;
	}
	return method->GetReturnType();
}

Type *TypeAnalysis::AnalyzeCall(const BuiltinMethod *method, const std::vector<Type *>& argumentTypes)
{
	// Check the argument count against method

	auto argumentCount = argumentTypes.size();
	auto parameterCount = method->GetParameterCount();
	if (parameterCount != BuiltinMethod::VariadicParameterCount && argumentCount != parameterCount)
	{
		TypeError(method, argumentTypes);
	}

	switch (method->GetKind())
	{
#define Require(x) if (!(x)) break

		case BuiltinMethod::Kind::Absolute:
		case BuiltinMethod::Kind::Ceiling:
		case BuiltinMethod::Kind::Floor:
		case BuiltinMethod::Kind::Round:
		{
			auto inputType = argumentTypes.at(0);
			Require(IsRealType(inputType));
			return inputType;
		}
		case BuiltinMethod::Kind::Negate:
		{
			auto inputType = argumentTypes.at(0);
			Require(IsRealType(inputType));
			if (IsBoolType(inputType))
			{
				return new BasicType(BasicType::Kind::Int16);
			}
			return inputType;
		}
		case BuiltinMethod::Kind::Conjugate:
		{
			auto inputType = argumentTypes.at(0);
			Require(IsComplexType(inputType));
			return inputType;
		}
		case BuiltinMethod::Kind::Reciprocal:
		{
			auto inputType = argumentTypes.at(0);
			Require(IsRealType(inputType));
			if (IsExtendedType(inputType))
			{
				return new BasicType(BasicType::Kind::Float64);
			}
			return new BasicType(BasicType::Kind::Float32);
		}
		case BuiltinMethod::Kind::Sign:
		{
			auto inputType = argumentTypes.at(0);
			Require(IsRealType(inputType));
			return new BasicType(BasicType::Kind::Int16);
		}
		case BuiltinMethod::Kind::Not:
		{
			auto inputType = argumentTypes.at(0);
			Require(IsBoolType(inputType));
			return inputType;
		}
		case BuiltinMethod::Kind::Pi:
		{
			auto inputType = argumentTypes.at(0);
			Require(IsNumericType(inputType));
			if (IsComplexType(inputType))
			{
				return inputType;
			}
			if (IsExtendedType(inputType))
			{
				return new BasicType(BasicType::Kind::Float64);
			}
			return new BasicType(BasicType::Kind::Float32);
		}
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
		{
			auto inputType = argumentTypes.at(0);
			Require(IsRealType(inputType));
			if (IsExtendedType(inputType))
			{
				return new BasicType(BasicType::Kind::Float64);
			}
			return new BasicType(BasicType::Kind::Float32);
		}
		case BuiltinMethod::Kind::Date:
		{
			auto inputType = argumentTypes.at(0);
			Require(IsDatetimeType(inputType));
			return new BasicType(BasicType::Kind::Date);
		}
		case BuiltinMethod::Kind::DateYear:
		case BuiltinMethod::Kind::DateMonth:
		{
			auto inputType = argumentTypes.at(0);
			Require(IsDatetimeType(inputType) || IsDateType(inputType) || IsMonthType(inputType));
			return new BasicType(BasicType::Kind::Int16);
		}
		case BuiltinMethod::Kind::DateDay:
		{
			auto inputType = argumentTypes.at(0);
			Require(IsDatetimeType(inputType) || IsDateType(inputType));
			return new BasicType(BasicType::Kind::Int16);
		}
		case BuiltinMethod::Kind::Time:
		{
			auto inputType = argumentTypes.at(0);
			Require(IsDatetimeType(inputType));
			return new BasicType(BasicType::Kind::Time);
		}
		case BuiltinMethod::Kind::TimeHour:
		case BuiltinMethod::Kind::TimeMinute:
		{
			auto inputType = argumentTypes.at(0);
			Require(IsDatetimeType(inputType) || IsMinuteType(inputType) || IsSecondType(inputType) || IsTimeType(inputType));
			return new BasicType(BasicType::Kind::Int16);
		}
		case BuiltinMethod::Kind::TimeSecond:
		{
			auto inputType = argumentTypes.at(0);
			Require(IsDatetimeType(inputType) || IsSecondType(inputType) || IsTimeType(inputType));
			return new BasicType(BasicType::Kind::Int16);
		}
		case BuiltinMethod::Kind::TimeMillisecond:
		{
			auto inputType = argumentTypes.at(0);
			Require(IsDatetimeType(inputType) || IsTimeType(inputType));
			return new BasicType(BasicType::Kind::Int16);
		}
		case BuiltinMethod::Kind::Less:
		case BuiltinMethod::Kind::Greater:
		case BuiltinMethod::Kind::LessEqual:
		case BuiltinMethod::Kind::GreaterEqual:
		{
			auto inputType0 = argumentTypes.at(0);
			auto inputType1 = argumentTypes.at(1);
			Require(IsOrderableTypes(inputType0, inputType1));
			return new BasicType(BasicType::Kind::Bool);
		}
		case BuiltinMethod::Kind::Equal:
		case BuiltinMethod::Kind::NotEqual:
		{
			auto inputType0 = argumentTypes.at(0);
			auto inputType1 = argumentTypes.at(1);
			Require(IsComparableTypes(inputType0, inputType1));
			return new BasicType(BasicType::Kind::Bool);
		}
		case BuiltinMethod::Kind::Plus:
		{
			auto inputType0 = argumentTypes.at(0);
			auto inputType1 = argumentTypes.at(1);
			if (IsRealType(inputType0) && IsRealType(inputType1))
			{
				return WidestType(inputType0, inputType1);
			}
			else if (IsIntegerType(inputType0) && IsCalendarType(inputType1))
			{
				return inputType1;
			}
			else if (IsCalendarType(inputType0) && IsIntegerType(inputType1))
			{
				return inputType0;
			}
			break;
		}
		case BuiltinMethod::Kind::Minus:
		{
			auto inputType0 = argumentTypes.at(0);
			auto inputType1 = argumentTypes.at(1);
			if (IsRealType(inputType0) && IsRealType(inputType1))
			{
				return WidestType(inputType0, inputType1);
			}
			else if (IsCalendarType(inputType0) && IsIntegerType(inputType1))
			{
				return inputType0;
			}
			else if (*inputType0 == *inputType1 && IsCalendarType(inputType0))
			{
				if (IsDatetimeType(inputType0))
				{
					return new BasicType(BasicType::Kind::Int64);
				}
				return new BasicType(BasicType::Kind::Int32);
			}
			break;
		}
		case BuiltinMethod::Kind::Multiply:
		case BuiltinMethod::Kind::Divide:
		{
			auto inputType0 = argumentTypes.at(0);
			auto inputType1 = argumentTypes.at(1);
			Require(IsNumericType(inputType0) && IsNumericType(inputType1));
			return WidestType(inputType0, inputType1);
		}
		case BuiltinMethod::Kind::Power:
		case BuiltinMethod::Kind::LogarithmBase:
		{
			auto inputType0 = argumentTypes.at(0);
			auto inputType1 = argumentTypes.at(1);
			Require(IsRealType(inputType0) && IsRealType(inputType1));
			auto widest = WidestType(inputType0, inputType1);
			if (IsExtendedType(widest))
			{
				return new BasicType(BasicType::Kind::Float64);
			}
			return new BasicType(BasicType::Kind::Float32);
		}
		case BuiltinMethod::Kind::Modulo:
		{
			auto inputType0 = argumentTypes.at(0);
			auto inputType1 = argumentTypes.at(1);
			Require(IsRealType(inputType0) && IsRealType(inputType1));
			if (IsBoolType(inputType0) && IsBoolType(inputType1))
			{
				return new BasicType(BasicType::Kind::Int16);
			}
			return WidestType(inputType0, inputType1);
		}
		case BuiltinMethod::Kind::And:
		case BuiltinMethod::Kind::Or:
		case BuiltinMethod::Kind::Nand:
		case BuiltinMethod::Kind::Nor:
		case BuiltinMethod::Kind::Xor:
		{
			auto inputType0 = argumentTypes.at(0);
			auto inputType1 = argumentTypes.at(1);
			Require(IsBoolType(inputType0) && IsBoolType(inputType1));
			return inputType0;
		}
		case BuiltinMethod::Kind::DatetimeDifference:
		{
			auto inputType0 = argumentTypes.at(0);
			auto inputType1 = argumentTypes.at(1);
			Require(*inputType0 == *inputType1 && IsCalendarType(inputType0));
			if (IsDatetimeType(inputType0))
			{
				return new BasicType(BasicType::Kind::Int64);
			}
			return new BasicType(BasicType::Kind::Int32);
		}
		case BuiltinMethod::Kind::Unique:
		{
			auto inputType = argumentTypes.at(0);
			Require(IsType<BasicType>(inputType));
			return new BasicType(BasicType::Kind::Int64);
		}
		case BuiltinMethod::Kind::String:
		{
			return new BasicType(BasicType::Kind::String);
		}
		case BuiltinMethod::Kind::Range:
		case BuiltinMethod::Kind::Factorial:
		{
			auto inputType = argumentTypes.at(0);
			Require(IsIntegerType(inputType));
			return new BasicType(BasicType::Kind::Int64);
		}
		case BuiltinMethod::Kind::Reverse:
		{
			auto inputType = argumentTypes.at(0);
			Require(IsType<BasicType>(inputType));
			return new BasicType(BasicType::Kind::Int64);
		}
		case BuiltinMethod::Kind::Where:
		{
			auto inputType = argumentTypes.at(0);
			if (IsBoolType(inputType))
			{
				return new BasicType(BasicType::Kind::Int64);
			}
			else if (auto listType = GetType<ListType>(inputType); listType != nullptr && IsBoolType(listType->GetElementType()))
			{
				return new ListType(new BasicType(BasicType::Kind::Int64));
			}
			break;
		}
		case BuiltinMethod::Kind::Group:
		{
			auto inputType = argumentTypes.at(0);
			Require(IsType<BasicType>(inputType) || IsType<ListType>(inputType));
			return new DictionaryType(
				new BasicType(BasicType::Kind::Int64),
				new ListType(new BasicType(BasicType::Kind::Int64))
			);
		}
		case BuiltinMethod::Kind::Append:
		{
			auto inputType0 = argumentTypes.at(0);
			auto inputType1 = argumentTypes.at(1);
			if (IsRealType(inputType0) && IsRealType(inputType1))
			{
				return WidestType(inputType0, inputType1);
			}
			else if (*inputType0 == *inputType1 && IsType<BasicType>(inputType0))
			{
				return inputType0;
			}
			else if (IsType<ListType>(inputType0) || IsType<ListType>(inputType1))
			{
				auto listType0 = GetType<ListType>(inputType0);
				auto listType1 = GetType<ListType>(inputType1);

				auto elementType0 = (listType0 == nullptr) ? inputType0 : listType0->GetElementType();
				auto elementType1 = (listType1 == nullptr) ? inputType1 : listType1->GetElementType();
				if (*elementType0 == *elementType1)
				{
					return new ListType(elementType0);
				}
				return new ListType(new BasicType(BasicType::Kind::Wildcard));
			}
			break;
		}
		case BuiltinMethod::Kind::Like:
		{
			auto inputType0 = argumentTypes.at(0);
			auto inputType1 = argumentTypes.at(1);
			Require(IsCharacterType(inputType0) && IsCharacterType(inputType1));
			return new BasicType(BasicType::Kind::Bool);
		}
		case BuiltinMethod::Kind::Compress:
		{
			auto predicateType = argumentTypes.at(0);
			auto dataType = argumentTypes.at(1);
			Require(IsBoolType(predicateType) && IsType<BasicType>(dataType));
			return dataType;
		}
		case BuiltinMethod::Kind::IndexOf:
		{
			auto inputType0 = argumentTypes.at(0);
			auto inputType1 = argumentTypes.at(1);
			Require(
				(IsRealType(inputType0) && IsRealType(inputType1)) ||
				(IsStringType(inputType0) && IsStringType(inputType1)) ||
				(IsSymbolType(inputType0) && IsSymbolType(inputType1))
			);
			return new BasicType(BasicType::Kind::Int64);
		}
		case BuiltinMethod::Kind::Order:
		{
			auto inputType0 = argumentTypes.at(0);
			auto inputType1 = argumentTypes.at(1);
			Require(
				IsRealType(inputType0) ||
				IsCharacterType(inputType0) ||
				IsCalendarType(inputType0) ||
				IsType<ListType>(inputType0)
			);
			Require(IsBoolType(inputType1));
			return new BasicType(BasicType::Kind::Int64);
		}
		case BuiltinMethod::Kind::Member:
		{
			auto inputType0 = argumentTypes.at(0);
			auto inputType1 = argumentTypes.at(1);
			Require(
				(IsRealType(inputType0) && IsRealType(inputType1)) ||
				(*inputType0 == *inputType1 && IsType<BasicType>(inputType0))
			);
			return new BasicType(BasicType::Kind::Bool);
		}
		case BuiltinMethod::Kind::Vector:
		{
			auto inputType0 = argumentTypes.at(0);
			auto inputType1 = argumentTypes.at(1);
			Require(IsIntegerType(inputType0) && (IsType<BasicType>(inputType1) || IsType<ListType>(inputType1)));
			return inputType1;
		}
		// @count and @len are aliases
		case BuiltinMethod::Kind::Length:
		case BuiltinMethod::Kind::Count:
		{
			return new BasicType(BasicType::Kind::Int64);
		}
		case BuiltinMethod::Kind::Sum:
		{
			auto inputType = argumentTypes.at(0);
			Require(IsRealType(inputType));
			if (IsFloatType(inputType))
			{
				return inputType;
			}
			if (IsBasicType(inputType, BasicType::Kind::Int64))
			{
				return new BasicType(BasicType::Kind::Int64);
			}
			return new BasicType(BasicType::Kind::Int32);
		}
		case BuiltinMethod::Kind::Average:
		{
			auto inputType = argumentTypes.at(0);
			Require(IsRealType(inputType));
			if (IsExtendedType(inputType))
			{
				return new BasicType(BasicType::Kind::Float64);
			}
			return new BasicType(BasicType::Kind::Float32);
		}
		case BuiltinMethod::Kind::Minimum:
		case BuiltinMethod::Kind::Maximum:
		{
			auto inputType = argumentTypes.at(0);
			Require(IsRealType(inputType));
			return inputType;
		}
		case BuiltinMethod::Kind::Raze:
		{
			auto inputType = argumentTypes.at(0);
			Require(IsType<ListType>(inputType));
			auto elementType = GetType<ListType>(inputType)->GetElementType();
			Require(IsType<BasicType>(elementType) && !IsBasicType(elementType, BasicType::Kind::Wildcard));
			return elementType;
		}
		case BuiltinMethod::Kind::List:
		{
			Type *elementType = nullptr;
			for (auto type : argumentTypes)
			{
				if (elementType == nullptr)
				{
					elementType = type;
				}
				else if (*elementType != *type)
				{
					elementType = new BasicType(BasicType::Kind::Wildcard);
					break;
				}
			}
			return new ListType(elementType);
		}
		case BuiltinMethod::Kind::Enlist:
		{
			return new ListType(argumentTypes.at(0));
		}
		case BuiltinMethod::Kind::ToList:
		{
			auto inputType = argumentTypes.at(0);
			Require(IsType<BasicType>(inputType));
			return new ListType(inputType);
		}
		case BuiltinMethod::Kind::Each:
		{
			auto inputType0 = argumentTypes.at(0);
			auto inputType1 = argumentTypes.at(1);
			Require(IsType<FunctionType>(inputType0) && IsType<ListType>(inputType1));

			auto functionType = GetType<FunctionType>(inputType0);
			auto listType = GetType<ListType>(inputType1);

			auto function = functionType->GetMethod();
			auto returnType = AnalyzeCall(function, {listType->GetElementType()});
			return new ListType(returnType);
		}
		case BuiltinMethod::Kind::EachItem:
		case BuiltinMethod::Kind::EachLeft:
		case BuiltinMethod::Kind::EachRight:
		{
			auto inputType0 = argumentTypes.at(0);
			auto inputType1 = argumentTypes.at(1);
			auto inputType2 = argumentTypes.at(2);
			Require(IsType<FunctionType>(inputType0));

			auto function = GetType<FunctionType>(inputType0)->GetMethod();

			auto unboxed1 = inputType1;
			auto unboxed2 = inputType2;
			bool unboxed = false;
			if (IsType<ListType>(unboxed1))
			{
				unboxed1 = GetType<ListType>(unboxed1)->GetElementType();
				unboxed = true;
			}
			if (IsType<ListType>(unboxed2))
			{
				unboxed2 = GetType<ListType>(unboxed2)->GetElementType();
				unboxed = true;
			}
		       	
			auto returnType = AnalyzeCall(function, {unboxed1, unboxed2});
			if (unboxed)
			{
				return new ListType(returnType);
			}
			return returnType;
		}
		case BuiltinMethod::Kind::Outer:
		{
			auto inputType0 = argumentTypes.at(0);
			auto inputType1 = argumentTypes.at(1);
			auto inputType2 = argumentTypes.at(2);
			Require(IsType<FunctionType>(inputType0) && IsType<BasicType>(inputType1) && IsType<BasicType>(inputType2));

			auto function = GetType<FunctionType>(inputType0)->GetMethod();
			auto returnType = AnalyzeCall(function, {inputType1, inputType2});

			return new ListType(returnType);
		}
		case BuiltinMethod::Kind::Enum:
		{
			auto inputType0 = argumentTypes.at(0);
			auto inputType1 = argumentTypes.at(1);
			Require(
				(IsType<BasicType>(inputType0) && IsType<BasicType>(inputType1)) ||
				(IsType<ListType>(inputType0) && IsType<ListType>(inputType1))
			);
				
			auto elementType0 = (IsType<BasicType>(inputType0)) ? inputType0 : GetType<ListType>(inputType0)->GetElementType();
			auto elementType1 = (IsType<BasicType>(inputType1)) ? inputType1 : GetType<ListType>(inputType1)->GetElementType();
			Require((IsRealType(elementType0) && IsRealType(elementType1)) || (IsType<BasicType>(elementType0) && *elementType0 == *elementType1));

			return new EnumerationType(inputType0, inputType1);
		}
		case BuiltinMethod::Kind::Dictionary:
		{
			auto inputType0 = argumentTypes.at(0);
			auto inputType1 = argumentTypes.at(1);
			return new DictionaryType(inputType0, inputType1);
		}
		case BuiltinMethod::Kind::Table:
		{
			auto inputType0 = argumentTypes.at(0);
			auto inputType1 = argumentTypes.at(1);
			Require(IsSymbolType(inputType0) && IsType<ListType>(inputType1));
			return new TableType();
		}
		case BuiltinMethod::Kind::KeyedTable:
		{
			auto inputType0 = argumentTypes.at(0);
			auto inputType1 = argumentTypes.at(1);
			Require(IsType<TableType>(inputType0) && IsType<TableType>(inputType1));
			return new KeyedTableType();
		}
		case BuiltinMethod::Kind::Keys:
		{
			auto inputType = argumentTypes.at(0);
			if (IsType<DictionaryType>(inputType))
			{
				return GetType<DictionaryType>(inputType)->GetKeyType();
			}
			else if (IsType<TableType>(inputType))
			{
				return GetType<TableType>(inputType)->GetKeyType();
			}
			else if (IsType<EnumerationType>(inputType))
			{
				return GetType<EnumerationType>(inputType)->GetKeyType();
			}
			else if (IsType<KeyedTableType>(inputType))
			{
				return new TableType();
			}
			break;
		}
		case BuiltinMethod::Kind::Values:
		{
			auto inputType = argumentTypes.at(0);
			if (IsType<DictionaryType>(inputType))
			{
				return GetType<DictionaryType>(inputType)->GetValueType();
			}
			else if (IsType<TableType>(inputType))
			{
				return GetType<TableType>(inputType)->GetValueType();
			}
			else if (IsType<EnumerationType>(inputType))
			{
				return GetType<EnumerationType>(inputType)->GetValueType();
			}
			else if (IsType<KeyedTableType>(inputType))
			{
				return new TableType();
			}
			break;
		}
		case BuiltinMethod::Kind::Meta:
		{
			auto inputType = argumentTypes.at(0);
			Require(IsType<TableType>(inputType) || IsType<KeyedTableType>(inputType));
			return new TableType();
		}
		case BuiltinMethod::Kind::Fetch:
		{
			auto inputType = argumentTypes.at(0);
			Require(IsType<EnumerationType>(inputType));
			return GetType<EnumerationType>(inputType)->GetValueType();
		}
		case BuiltinMethod::Kind::ColumnValue:
		{
			auto inputType0 = argumentTypes.at(0);
			auto inputType1 = argumentTypes.at(1);
			Require(IsType<TableType>(inputType0) && IsSymbolType(inputType1));

			// A column value call is intentionally untyped since it comes from the runtime system.
			// It must be cast before assigning to a variable

			return nullptr;
		}
		case BuiltinMethod::Kind::LoadTable:
		{
			auto inputType = argumentTypes.at(0);
			Require(IsSymbolType(inputType));
			return new TableType();
		}
		case BuiltinMethod::Kind::DatetimeAdd:
		case BuiltinMethod::Kind::DatetimeSubtract:
		{
			auto inputType0 = argumentTypes.at(0);
			auto inputType1 = argumentTypes.at(1);
			auto inputType2 = argumentTypes.at(2);
			Require(IsCalendarType(inputType0) && IsIntegerType(inputType1) && IsSymbolType(inputType2));
			return inputType0;
		}
		case BuiltinMethod::Kind::JoinIndex:
		{
			auto inputType0 = argumentTypes.at(0);
			auto inputType1 = argumentTypes.at(1);
			auto inputType2 = argumentTypes.at(2);
			Require(IsType<FunctionType>(inputType0) &&
				(IsType<BasicType>(inputType1) && IsType<BasicType>(inputType2)) ||
				(IsType<ListType>(inputType1) && IsType<ListType>(inputType2))
			);

			auto function = GetType<FunctionType>(inputType0)->GetMethod();
			auto returnType = AnalyzeCall(function, {inputType1, inputType2});
			Require(IsBoolType(returnType));

			return new ListType(new BasicType(BasicType::Kind::Int64));
		}
		case BuiltinMethod::Kind::Index:
		{
			auto inputType0 = argumentTypes.at(0);
			auto inputType1 = argumentTypes.at(1);
			Require(IsType<BasicType>(inputType0));
			if (IsIntegerType(inputType1))
			{
				return inputType0;
			}
			else if (auto listType = GetType<ListType>(inputType1); listType != nullptr && IsIntegerType(listType->GetElementType()))
			{
				return new ListType(inputType0);
			}
			break;
		}
		case BuiltinMethod::Kind::IndexAssignment:
		{
			auto inputType0 = argumentTypes.at(0);
			auto inputType1 = argumentTypes.at(1);
			auto inputType2 = argumentTypes.at(2);
			Require(IsType<BasicType>(inputType0) && IsIntegerType(inputType1) && IsAssignableType(inputType0, inputType2));
			return inputType0;
		}
		case BuiltinMethod::Kind::SubString:
		{
			auto inputType0 = argumentTypes.at(0);
			auto inputType1 = argumentTypes.at(1);
			auto inputType2 = argumentTypes.at(2);
			Require(IsStringType(inputType0) && IsIntegerType(inputType1) && IsIntegerType(inputType2));
			return inputType0;
		}
		default:
		{
			Utils::Logger::LogError("Type analysis does not support builtin method '" + method->GetName() + "'");
		}
	}
	
	// If we cannot infer a return type, report a type error

	TypeError(method, argumentTypes);
}

void TypeAnalysis::Visit(CastExpression *cast)
{
	// Visit the child expression and type

	ForwardTraversal::Visit(cast);

	// The input value of a cast may not yet have a type, in which case we defer
	// checking to the interpreter. Otherwise perform a static check

	auto expressionType = cast->GetExpression()->GetType();
	auto castType = cast->GetCastType();

	if (expressionType == nullptr || *expressionType == *castType)
	{
		cast->SetType(cast->GetCastType());
	}
	else
	{
		Utils::Logger::LogError("Invalid cast, '" + expressionType->ToString() + "' cannot be cast to '" + castType->ToString() + "'");
	}
}

void TypeAnalysis::Visit(ReturnStatement *ret)
{
	// Visit the returned identifier

	ForwardTraversal::Visit(ret);

	// Check the current function return type against that of the identifier

	// The input value of a cast may not yet have a type, in which case we defer
	// checking to the interpreter. Otherwise perform a static check

	auto returnType = m_currentMethod->GetReturnType();
	auto identifierType = ret->GetIdentifier()->GetType();

	if (*returnType != *identifierType)
	{
		Utils::Logger::LogError("Method '" + m_currentMethod->GetName() + "' expects a return value of type '" + returnType->ToString() + "' but received '" + identifierType->ToString() + "'");
	}
}

void TypeAnalysis::Visit(Identifier *identifier)
{
	identifier->SetType(identifier->GetDeclaration()->GetType());
}

void TypeAnalysis::Visit(FunctionLiteral *literal)
{
	// Propagate the method from the literal expression to the expression type

	auto type = GetType<FunctionType>(literal->GetType());
	if (type == nullptr)
	{
		Utils::Logger::LogError("Invalid type '" + literal->GetType()->ToString() + "' for function literal '" + literal->ToString() + "'");
	}
	type->SetMethod(literal->GetMethod());
}

void TypeAnalysis::Visit(EnumerationType *type)
{
	ForwardTraversal::Visit(type);

	auto keyType = type->GetKeyType();
	auto valueType = type->GetValueType();
	
	if ((IsType<BasicType>(keyType) && IsType<BasicType>(valueType)) ||
		(IsType<ListType>(keyType) && IsType<ListType>(valueType)))
	{
		auto unboxedKeyType = (IsType<BasicType>(keyType)) ? keyType : GetType<ListType>(keyType)->GetElementType();
		auto unboxedValueType = (IsType<BasicType>(valueType)) ? valueType : GetType<ListType>(valueType)->GetElementType();

		if ((IsRealType(unboxedKeyType) && IsRealType(unboxedValueType)) || (IsType<BasicType>(unboxedKeyType) && *unboxedKeyType == *unboxedValueType))
		{
			return;
		}
	}

	Utils::Logger::LogError("Invalid key/value types for enumeration '" + type->ToString() + "'");
}

}
