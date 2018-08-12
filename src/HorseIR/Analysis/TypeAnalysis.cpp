#include "HorseIR/Analysis/TypeAnalysis.h"

#include "HorseIR/TypeUtils.h"
#include "HorseIR/Tree/BuiltinMethod.h"
#include "HorseIR/Tree/Method.h"
#include "HorseIR/Tree/Program.h"
#include "HorseIR/Tree/Expressions/CallExpression.h"
#include "HorseIR/Tree/Expressions/CastExpression.h"
#include "HorseIR/Tree/Expressions/Expression.h"
#include "HorseIR/Tree/Expressions/Identifier.h"
#include "HorseIR/Tree/Expressions/Literals/FunctionLiteral.h"
#include "HorseIR/Tree/Statements/AssignStatement.h"
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
	auto arguments = call->GetArguments();
	std::vector<Type *> argumentTypes;
	for (auto argument : arguments)
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
			//TODO: Analyze arguments
			return static_cast<const Method *>(method)->GetReturnType();
		default:
			Utils::Logger::LogError("Unsupported method kind");
	}
}

[[noreturn]] void TypeError(const BuiltinMethod *method, const std::vector<Type *>& argumentTypes)
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
	message += "] to builtin function '" + method->GetName() + "'";
	Utils::Logger::LogError(message);
}

Type *TypeAnalysis::AnalyzeCall(const BuiltinMethod *method, const std::vector<Type *>& argumentTypes)
{
	// Check the argument count against method

	auto argumentCount = argumentTypes.size();
	auto parameterCount = method->GetParameterCount();
	if (argumentCount != parameterCount)
	{
		TypeError(method, argumentTypes);
	}

	switch (method->GetKind())
	{
#define RequireType(x) if (!(x)) break

		case BuiltinMethod::Kind::Absolute:
		case BuiltinMethod::Kind::Ceiling:
		case BuiltinMethod::Kind::Floor:
		case BuiltinMethod::Kind::Round:
		{
			auto inputType = argumentTypes.at(0);
			RequireType(IsRealType(inputType));
			return inputType;
		}
		case BuiltinMethod::Kind::Negate:
		{
			auto inputType = argumentTypes.at(0);
			RequireType(IsRealType(inputType));
			if (IsBoolType(inputType))
			{
				return new BasicType(BasicType::Kind::Int16);
			}
			return inputType;
		}
		case BuiltinMethod::Kind::Sign:
		{
			auto inputType = argumentTypes.at(0);
			RequireType(IsRealType(inputType));
			return new BasicType(BasicType::Kind::Int16);
		}
		case BuiltinMethod::Kind::Not:
		{
			auto inputType = argumentTypes.at(0);
			RequireType(IsBoolType(inputType));
			return inputType;
		}
		case BuiltinMethod::Kind::Reciprocal:
		case BuiltinMethod::Kind::Pi:
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
			RequireType(IsRealType(inputType));
			if (IsExtendedType(inputType))
			{
				return new BasicType(BasicType::Kind::Float64);
			}
			return new BasicType(BasicType::Kind::Float32);
		}
		case BuiltinMethod::Kind::Less:
		case BuiltinMethod::Kind::Greater:
		case BuiltinMethod::Kind::LessEqual:
		case BuiltinMethod::Kind::GreaterEqual:
		case BuiltinMethod::Kind::Equal:
		case BuiltinMethod::Kind::NotEqual:
		{
			auto inputType0 = argumentTypes.at(0);
			auto inputType1 = argumentTypes.at(1);
			RequireType(IsComparableTypes(inputType0, inputType1));
			return new BasicType(BasicType::Kind::Bool);
		}
		case BuiltinMethod::Kind::Plus:
		case BuiltinMethod::Kind::Minus:
		case BuiltinMethod::Kind::Multiply:
		case BuiltinMethod::Kind::Divide:
		{
			auto inputType0 = argumentTypes.at(0);
			auto inputType1 = argumentTypes.at(1);
			RequireType(IsRealType(inputType0) && IsRealType(inputType1));
			return WidestType(inputType0, inputType1);
		}
		case BuiltinMethod::Kind::Power:
		case BuiltinMethod::Kind::LogarithmBase:
		{
			auto inputType0 = argumentTypes.at(0);
			auto inputType1 = argumentTypes.at(1);
			RequireType(IsRealType(inputType0) && IsRealType(inputType1));
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
			RequireType(IsRealType(inputType0) && IsRealType(inputType1));
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
			RequireType(IsBoolType(inputType0) && IsBoolType(inputType1));
			return inputType0;
		}
		case BuiltinMethod::Kind::Unique:
		{
			auto inputType = argumentTypes.at(0);
			RequireType(IsBasicType(inputType));
			return new BasicType(BasicType::Kind::Int64);
		}
		case BuiltinMethod::Kind::Where:
		{
			auto inputType = argumentTypes.at(0);
			if (IsListType(inputType) && IsBoolType(static_cast<const ListType *>(inputType)->GetElementType()))
			{
				return new ListType(new BasicType(BasicType::Kind::Int64));
			}
			else if (IsBoolType(inputType))
			{
				return new BasicType(BasicType::Kind::Int64);
			}
			break;
		}
		case BuiltinMethod::Kind::Group:
		{
			auto inputType = argumentTypes.at(0);
			if (IsBasicType(inputType) || IsListType(inputType))
			{
				return new DictionaryType(
					new BasicType(BasicType::Kind::Int64),
					new ListType(new BasicType(BasicType::Kind::Int64))
				);
			}
			break;
		}
		case BuiltinMethod::Kind::Append:
		{
			auto inputType0 = argumentTypes.at(0);
			auto inputType1 = argumentTypes.at(1);
			if (IsRealType(inputType0) && IsRealType(inputType1))
			{
				return WidestType(inputType0, inputType1);
			}
			//TODO: List cases
			break;
		}
		case BuiltinMethod::Kind::Like:
		{
			auto inputType0 = argumentTypes.at(0);
			auto inputType1 = argumentTypes.at(1);
			RequireType(IsStringType(inputType0) && IsStringType(inputType1));
			return new BasicType(BasicType::Kind::Bool);
		}
		case BuiltinMethod::Kind::Compress:
		{
			auto predicateType = argumentTypes.at(0);
			auto dataType = argumentTypes.at(1);
			RequireType(IsBoolType(predicateType) && IsBasicType(dataType));
			return dataType;
		}
		case BuiltinMethod::Kind::IndexOf:
		{
			auto inputType0 = argumentTypes.at(0);
			auto inputType1 = argumentTypes.at(1);
			RequireType(
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
			RequireType(IsBoolType(inputType1));
			if (IsBasicType(inputType0) && !IsComplexType(inputType0))
			{
				return new BasicType(BasicType::Kind::Int32);
			}
			else if (IsListType(inputType1))
			{
				//TODO: List cases
			}
			break;
		}
		case BuiltinMethod::Kind::Member:
		{
			auto inputType0 = argumentTypes.at(0);
			auto inputType1 = argumentTypes.at(1);
			RequireType(
				(IsRealType(inputType0) && IsRealType(inputType1)) ||
				(IsBasicType(inputType0) && IsBasicType(inputType1) && *inputType0 == *inputType1)
			);
			return new BasicType(BasicType::Kind::Bool);
		}
		case BuiltinMethod::Kind::Vector:
		{
			auto inputType0 = argumentTypes.at(0);
			auto inputType1 = argumentTypes.at(1);
			RequireType(IsIntegerType(inputType0) && (IsBasicType(inputType1) || IsListType(inputType1)));
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
			RequireType(IsRealType(inputType));
			if (IsFloatType(inputType))
			{
				return inputType;
			}
			return new BasicType(BasicType::Kind::Int64);
		}
		case BuiltinMethod::Kind::Average:
		{
			auto inputType = argumentTypes.at(0);
			RequireType(IsRealType(inputType));
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
			RequireType(IsRealType(inputType));
			return inputType;
		}
		case BuiltinMethod::Kind::Raze:
		{
			auto inputType = argumentTypes.at(0);
			RequireType(IsListType(inputType));
			return static_cast<const ListType *>(inputType)->GetElementType();
		}
		case BuiltinMethod::Kind::Enlist:
		{
			return new ListType(argumentTypes.at(0));
		}
		case BuiltinMethod::Kind::ToList:
		{
			auto inputType = argumentTypes.at(0);
			RequireType(IsBasicType(inputType));
			return new ListType(inputType);
		}
		case BuiltinMethod::Kind::Each:
		{
			auto inputType0 = argumentTypes.at(0);
			auto inputType1 = argumentTypes.at(1);
			RequireType(IsFunctionType(inputType0) && IsListType(inputType1));

			auto function = static_cast<FunctionType *>(inputType0)->GetMethod();
			auto returnType = AnalyzeCall(function, {static_cast<ListType *>(inputType1)->GetElementType()});
			return new ListType(returnType);
		}
		case BuiltinMethod::Kind::EachItem:
		{
			auto inputType0 = argumentTypes.at(0);
			auto inputType1 = argumentTypes.at(1);
			auto inputType2 = argumentTypes.at(2);
			RequireType(
				IsFunctionType(inputType0) && (
					(IsBasicType(inputType1) || IsListType(inputType1)) ||
					(IsBasicType(inputType2) || IsListType(inputType2))
				)
			);

			auto function = static_cast<FunctionType *>(inputType0)->GetMethod();

			auto unboxed1 = inputType1;
			auto unboxed2 = inputType2;
			bool unboxed = false;
			if (IsListType(unboxed1))
			{
				unboxed1 = static_cast<ListType *>(unboxed1)->GetElementType();
				unboxed = true;
			}
			if (IsListType(unboxed2))
			{
				unboxed2 = static_cast<ListType *>(unboxed2)->GetElementType();
				unboxed = true;
			}
		       	
			auto returnType = AnalyzeCall(function, {unboxed1, unboxed2});
			if (unboxed)
			{
				return new ListType(returnType);
			}
			return returnType;
		}
		case BuiltinMethod::Kind::EachLeft:
		case BuiltinMethod::Kind::EachRight:
		{
			//TODO: Add each support
			break;
		}
		case BuiltinMethod::Kind::Outer:
		{
			//TODO: Add outer product support
			break;
		}
		case BuiltinMethod::Kind::Enum:
		{
			//TODO: Add @enum type rules
			break;
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
			RequireType(IsSymbolType(inputType0) && IsListType(inputType1));
			return new TableType();
		}
		case BuiltinMethod::Kind::Keys:
		case BuiltinMethod::Kind::Values:
		{
			//TODO: Database functions
			break;
		}
		case BuiltinMethod::Kind::ColumnValue:
		{
			auto inputType0 = argumentTypes.at(0);
			auto inputType1 = argumentTypes.at(1);
			RequireType(IsTableType(inputType0) && IsSymbolType(inputType1));

			// A column value call is intentionally untyped since it comes from the runtime system.
			// It must be cast before assigning to a variable

			return nullptr;
		}
		case BuiltinMethod::Kind::LoadTable:
		{
			auto inputType = argumentTypes.at(0);
			RequireType(IsSymbolType(inputType));
			return new TableType();
		}
		case BuiltinMethod::Kind::Fetch:
		{
			//TODO: @fetch type rules
			break;
		}
		case BuiltinMethod::Kind::Index:
		{
			auto inputType0 = argumentTypes.at(0);
			auto inputType1 = argumentTypes.at(1);
			RequireType(IsBasicType(inputType0));
			if (IsIntegerType(inputType1))
			{
				return inputType0;
			}
			else if (IsListType(inputType1) && IsIntegerType(static_cast<const ListType *>(inputType1)->GetElementType()))
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
			RequireType(IsBasicType(inputType0) && IsIntegerType(inputType1) && *inputType0 == *inputType2);
			return inputType0;
		}
		default:
			Utils::Logger::LogError("Type analysis does not support builtin method " + method->GetName());
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
		Utils::Logger::LogError("Invalid cast, " + expressionType->ToString() + " cannot be cast to " + castType->ToString());
	}
}

void TypeAnalysis::Visit(Identifier *identifier)
{
	identifier->SetType(identifier->GetDeclaration()->GetType());
}

void TypeAnalysis::Visit(FunctionLiteral *literal)
{
	auto type = static_cast<FunctionType *>(literal->GetType());
	type->SetMethod(literal->GetMethod());
}

}
