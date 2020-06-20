#include "HorseIR/Semantics/TypeChecker.h"

#include <algorithm>

#include "HorseIR/Tree/Tree.h"
#include "HorseIR/Utils/TypeUtils.h"
#include "HorseIR/Utils/PrettyPrinter.h"

#include "Utils/Logger.h"

namespace HorseIR {

void TypeChecker::Analyze(Program *program)
{
	program->Accept(*this);
}

void TypeChecker::VisitOut(GlobalDeclaration *global)
{
	// Check the expression type matches the declaration type

	const auto& expressionTypes = global->GetExpression()->GetTypes();
	const auto declarationType = global->GetDeclaration()->GetType();
	if (!TypeUtils::IsTypesAssignable({declarationType}, expressionTypes))
	{
		Utils::Logger::LogError("Expression type " + TypeUtils::TypeString(expressionTypes) + " does not match declaration type " + TypeUtils::TypeString(declarationType));
	}
}

bool TypeChecker::VisitIn(Function *function)
{
	// Store the current function for checking the return type

	m_currentFunction = function;
	return true;
}

void TypeChecker::VisitOut(Function *function)
{
	m_currentFunction = nullptr;
}

void TypeChecker::VisitOut(AssignStatement *assign)
{
	// Assemble the expression and target LValue types

	std::vector<Type *> targetTypes;
	for (const auto target : assign->GetTargets())
	{
		targetTypes.push_back(target->GetType());
	}
	const auto& expressionTypes = assign->GetExpression()->GetTypes();

	// Check that the expression and destination types match, allowing for runtime checks

	if (!TypeUtils::IsTypesAssignable(targetTypes, expressionTypes))
	{
		Utils::Logger::LogError("Expression type " + TypeUtils::TypeString(expressionTypes) + " does not match destination type " + TypeUtils::TypeString(targetTypes));
	}

	auto index = 0u;
	for (const auto target : assign->GetTargets())
	{
		const auto symbol = target->GetSymbol();
		if (auto node = dynamic_cast<VariableDeclaration *>(symbol->node))
		{
			if (TypeUtils::IsType<WildcardType>(target->GetType()))
			{
				node->SetType(expressionTypes.at(index++));
			}
		}
		else
		{
			Utils::Logger::LogError("Assignment target expects variable kind");
		}
	}
}

void TypeChecker::VisitOut(ExpressionStatement *expressionS)
{
	// Check the expression in the statement has no type (void only)

	const auto& expressionTypes = expressionS->GetExpression()->GetTypes();
	if (!TypeUtils::IsEmptyType(expressionTypes))
	{
		Utils::Logger::LogError("Expression statement must have no type, received " + TypeUtils::TypeString(expressionTypes));
	}
}

void TypeChecker::VisitOut(IfStatement *ifS)
{
	// Check the condition is a boolean type

	const auto conditionType = ifS->GetCondition()->GetType();
	if (!TypeUtils::IsBooleanType(conditionType))
	{
		Utils::Logger::LogError("If condition must be a boolean type, received " + TypeUtils::TypeString(conditionType));
	}
}

void TypeChecker::VisitOut(WhileStatement *whileS)
{
	// Check the condition is a boolean type

	const auto conditionType = whileS->GetCondition()->GetType();
	if (!TypeUtils::IsBooleanType(conditionType))
	{
		Utils::Logger::LogError("While condition must be a boolean type, received " + TypeUtils::TypeString(conditionType));
	}
}

void TypeChecker::VisitOut(RepeatStatement *repeatS)
{
	// Check the condition is an integer type

	const auto conditionType = repeatS->GetCondition()->GetType();
	if (!TypeUtils::IsIntegerType(conditionType))
	{
		Utils::Logger::LogError("Repeat condition must be an integer type, received " + TypeUtils::TypeString(conditionType));
	}
}

void TypeChecker::VisitOut(ReturnStatement *ret)
{
	// Check the current function return type against that of the operands
	//
	// The input value of a cast may not yet have a type, in which case we defer
	// checking to the interpreter. Otherwise perform a static check

	std::vector<Type *> operandTypes;
	for (const auto operand : ret->GetOperands())
	{
		operandTypes.push_back(operand->GetType());
	}
	const auto& returnTypes = m_currentFunction->GetReturnTypes();

	// Check that the types are equal, allowing for runtime checks

	if (!TypeUtils::IsTypesAssignable(returnTypes, operandTypes))
	{
		Utils::Logger::LogError("Function '" + m_currentFunction->GetName() + "' expected return type " + TypeUtils::TypeString(returnTypes) + ", received " + TypeUtils::TypeString(operandTypes));
	}
}

void TypeChecker::VisitOut(CastExpression *cast)
{
	// The input value of a cast may not yet have a type, in which case we defer
	// checking to the interpreter. Otherwise perform a static check

	const auto& expressionTypes = cast->GetExpression()->GetTypes();
	const auto castType = cast->GetCastType();
	if (TypeUtils::IsSingleType(expressionTypes))
	{
		const auto expressionType = TypeUtils::GetSingleType(expressionTypes);
		if (TypeUtils::IsType<WildcardType>(expressionType) || TypeUtils::IsCastable(castType, expressionType))
		{
			cast->SetTypes({castType});
			return;
		}
	}

	Utils::Logger::LogError("Invalid cast, " + TypeUtils::TypeString(expressionTypes) + " cannot be cast to " + TypeUtils::TypeString(castType));
}

void TypeChecker::VisitOut(CallExpression *call)
{
	const auto function = call->GetFunctionLiteral()->GetFunction();
	const auto functionType = TypeUtils::GetType<FunctionType>(call->GetFunctionLiteral()->GetType());

	// Construct the list of argument types, decomposing lists (for future compatibility), and checking void

	std::vector<Type *> argumentTypes;
	for (const auto argument : call->GetArguments())
	{
		argumentTypes.push_back(argument->GetType());
	}

	// Analyze the function according to the type rules and form the return types

	const auto returnTypes = AnalyzeCall(functionType, argumentTypes);
	call->SetTypes(returnTypes);
}

std::vector<Type *> TypeChecker::AnalyzeCall(const FunctionType *functionType, const std::vector<Type *>& argumentTypes) const
{
	switch (functionType->GetFunctionKind())
	{
		case FunctionType::FunctionKind::Builtin:
			return AnalyzeCall(static_cast<const BuiltinFunction *>(functionType->GetFunctionDeclaration()), argumentTypes);
		case FunctionType::FunctionKind::Definition:
			return AnalyzeCall(static_cast<const Function *>(functionType->GetFunctionDeclaration()), functionType, argumentTypes);
		default:
			Utils::Logger::LogError("Unsupported function type");
	}
}

[[noreturn]] void TypeChecker::TypeError(const FunctionDeclaration *function, const std::vector<Type *>& argumentTypes) const
{
	Utils::Logger::LogError("Incompatible arguments " + TypeUtils::TypeString(argumentTypes) + " for function '" + function->GetName() + "'");
}

std::vector<Type *> TypeChecker::AnalyzeCall(const Function *function, const FunctionType *functionType, const std::vector<Type *>& argumentTypes) const
{
	// Check the arguments are equal with the parameters, allowing for runtime checks
	
	if (!TypeUtils::IsTypesAssignable(functionType->GetParameterTypes(), argumentTypes))
	{
		TypeError(function, argumentTypes);
	}
	return functionType->GetReturnTypes();
}

std::vector<Type *> TypeChecker::AnalyzeCall(const BuiltinFunction *function, const std::vector<Type *>& argumentTypes) const
{
	// Check the argument count against function

	auto argumentCount = argumentTypes.size();
	auto parameterCount = function->GetParameterCount();
	if (parameterCount != BuiltinFunction::VariadicParameterCount && argumentCount != parameterCount)
	{
		TypeError(function, argumentTypes);
	}

	switch (function->GetPrimitive())
	{
#define Require(x) if (!(x)) break

		// Unary
		case BuiltinFunction::Primitive::Absolute:
		case BuiltinFunction::Primitive::Ceiling:
		case BuiltinFunction::Primitive::Floor:
		case BuiltinFunction::Primitive::Round:
		{
			const auto inputType = argumentTypes.at(0);
			Require(TypeUtils::IsRealType(inputType));
			return {inputType};
		}
		case BuiltinFunction::Primitive::Negate:
		{
			const auto inputType = argumentTypes.at(0);
			Require(TypeUtils::IsRealType(inputType));
			if (TypeUtils::IsBooleanType(inputType))
			{
				return {new BasicType(BasicType::BasicKind::Int16)};
			}
			return {inputType};
		}
		case BuiltinFunction::Primitive::Conjugate:
		{
			const auto inputType = argumentTypes.at(0);
			Require(TypeUtils::IsComplexType(inputType));
			return {inputType};
		}
		case BuiltinFunction::Primitive::Reciprocal:
		{
			const auto inputType = argumentTypes.at(0);
			Require(TypeUtils::IsRealType(inputType));
			if (TypeUtils::IsExtendedType(inputType))
			{
				return {new BasicType(BasicType::BasicKind::Float64)};
			}
			return {new BasicType(BasicType::BasicKind::Float32)};
		}
		case BuiltinFunction::Primitive::Sign:
		{
			const auto inputType = argumentTypes.at(0);
			Require(TypeUtils::IsRealType(inputType));
			return {new BasicType(BasicType::BasicKind::Int16)};
		}
		case BuiltinFunction::Primitive::Pi:
		{
			const auto inputType = argumentTypes.at(0);
			Require(TypeUtils::IsNumericType(inputType));
			if (TypeUtils::IsComplexType(inputType))
			{
				return {inputType};
			}
			if (TypeUtils::IsExtendedType(inputType))
			{
				return {new BasicType(BasicType::BasicKind::Float64)};
			}
			return {new BasicType(BasicType::BasicKind::Float32)};
		}
		case BuiltinFunction::Primitive::Not:
		{
			const auto inputType = argumentTypes.at(0);
			Require(TypeUtils::IsBooleanType(inputType));
			return {inputType};
		}
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
		{
			const auto inputType = argumentTypes.at(0);
			Require(TypeUtils::IsRealType(inputType));
			if (TypeUtils::IsExtendedType(inputType))
			{
				return {new BasicType(BasicType::BasicKind::Float64)};
			}
			return {new BasicType(BasicType::BasicKind::Float32)};
		}

		// Binary
		case BuiltinFunction::Primitive::Less:
		case BuiltinFunction::Primitive::Greater:
		case BuiltinFunction::Primitive::LessEqual:
		case BuiltinFunction::Primitive::GreaterEqual:
		{
			const auto inputType0 = argumentTypes.at(0);
			const auto inputType1 = argumentTypes.at(1);
			Require(TypeUtils::IsOrderableTypes(inputType0, inputType1));
			return {new BasicType(BasicType::BasicKind::Boolean)};
		}
		case BuiltinFunction::Primitive::Equal:
		case BuiltinFunction::Primitive::NotEqual:
		{
			const auto inputType0 = argumentTypes.at(0);
			const auto inputType1 = argumentTypes.at(1);
			Require(TypeUtils::IsComparableTypes(inputType0, inputType1));
			return {new BasicType(BasicType::BasicKind::Boolean)};
		}
		case BuiltinFunction::Primitive::Plus:
		{
			const auto inputType0 = argumentTypes.at(0);
			const auto inputType1 = argumentTypes.at(1);
			if (TypeUtils::IsRealType(inputType0) && TypeUtils::IsRealType(inputType1))
			{
				return {TypeUtils::WidestType(inputType0, inputType1)};
			}
			else if (TypeUtils::IsIntegerType(inputType0) && TypeUtils::IsCalendarType(inputType1))
			{
				return {inputType1};
			}
			else if (TypeUtils::IsCalendarType(inputType0) && TypeUtils::IsIntegerType(inputType1))
			{
				return {inputType0};
			}
			break;
		}
		case BuiltinFunction::Primitive::Minus:
		{
			const auto inputType0 = argumentTypes.at(0);
			const auto inputType1 = argumentTypes.at(1);
			if (TypeUtils::IsRealType(inputType0) && TypeUtils::IsRealType(inputType1))
			{
				return {TypeUtils::WidestType(inputType0, inputType1)};
			}
			else if (TypeUtils::IsCalendarType(inputType0) && TypeUtils::IsIntegerType(inputType1))
			{
				return {inputType0};
			}
			else if (TypeUtils::IsTypesEqual(inputType0, inputType1) && TypeUtils::IsCalendarType(inputType0))
			{
				if (TypeUtils::IsDatetimeType(inputType0))
				{
					return {new BasicType(BasicType::BasicKind::Int64)};
				}
				return {new BasicType(BasicType::BasicKind::Int32)};
			}
			break;
		}
		case BuiltinFunction::Primitive::Multiply:
		case BuiltinFunction::Primitive::Divide:
		{
			const auto inputType0 = argumentTypes.at(0);
			const auto inputType1 = argumentTypes.at(1);
			Require(TypeUtils::IsNumericType(inputType0) && TypeUtils::IsNumericType(inputType1));
			return {TypeUtils::WidestType(inputType0, inputType1)};
		}
		case BuiltinFunction::Primitive::Power:
		case BuiltinFunction::Primitive::LogarithmBase:
		{
			const auto inputType0 = argumentTypes.at(0);
			const auto inputType1 = argumentTypes.at(1);
			Require(TypeUtils::IsRealType(inputType0) && TypeUtils::IsRealType(inputType1));
			const auto widest = TypeUtils::WidestType(inputType0, inputType1);
			if (TypeUtils::IsExtendedType(widest))
			{
				return {new BasicType(BasicType::BasicKind::Float64)};
			}
			return {new BasicType(BasicType::BasicKind::Float32)};
		}
		case BuiltinFunction::Primitive::Modulo:
		{
			const auto inputType0 = argumentTypes.at(0);
			const auto inputType1 = argumentTypes.at(1);
			Require(TypeUtils::IsRealType(inputType0) && TypeUtils::IsRealType(inputType1));
			if (TypeUtils::IsBooleanType(inputType0) && TypeUtils::IsBooleanType(inputType1))
			{
				return {new BasicType(BasicType::BasicKind::Int16)};
			}
			return {TypeUtils::WidestType(inputType0, inputType1)};
		}
		case BuiltinFunction::Primitive::And:
		case BuiltinFunction::Primitive::Or:
		case BuiltinFunction::Primitive::Nand:
		case BuiltinFunction::Primitive::Nor:
		case BuiltinFunction::Primitive::Xor:
		{
			const auto inputType0 = argumentTypes.at(0);
			const auto inputType1 = argumentTypes.at(1);
			Require(TypeUtils::IsBooleanType(inputType0) && TypeUtils::IsBooleanType(inputType1));
			return {inputType0};
		}

		// Algebraic Unary
		case BuiltinFunction::Primitive::Unique:
		{
			const auto inputType = argumentTypes.at(0);
			Require(TypeUtils::IsType<BasicType>(inputType));
			return {new BasicType(BasicType::BasicKind::Int64)};
		}
		case BuiltinFunction::Primitive::Range:
		case BuiltinFunction::Primitive::Factorial:
		case BuiltinFunction::Primitive::Seed:
		case BuiltinFunction::Primitive::Random:
		{
			const auto inputType = argumentTypes.at(0);
			Require(TypeUtils::IsIntegerType(inputType));
			return {new BasicType(BasicType::BasicKind::Int64)};
		}
		case BuiltinFunction::Primitive::Flip:
		{
			const auto inputType = argumentTypes.at(0);
			Require(TypeUtils::IsType<ListType>(inputType));
			return {inputType};
		}
		case BuiltinFunction::Primitive::Reverse:
		{
			const auto inputType = argumentTypes.at(0);
			Require(TypeUtils::IsType<BasicType>(inputType));
			return {inputType};
		}
		case BuiltinFunction::Primitive::Where:
		{
			const auto inputType = argumentTypes.at(0);
			Require(TypeUtils::IsBooleanType(inputType));
			return {new BasicType(BasicType::BasicKind::Int64)};
		}
		case BuiltinFunction::Primitive::Group:
		{
			const auto inputType = argumentTypes.at(0);
			Require(TypeUtils::IsType<BasicType>(inputType) || TypeUtils::IsType<ListType>(inputType));
			if (const auto listType = TypeUtils::GetType<ListType>(inputType))
			{
				// All cells must be boolean type

				Require(TypeUtils::ForallElements(listType, TypeUtils::IsType<BasicType>));
			}
			return {new DictionaryType(new BasicType(BasicType::BasicKind::Int64), new BasicType(BasicType::BasicKind::Int64))};
		}

		// Algebraic Binary
		case BuiltinFunction::Primitive::Append:
		{
			const auto inputType0 = argumentTypes.at(0);
			const auto inputType1 = argumentTypes.at(1);

			if (TypeUtils::IsRealType(inputType0) && TypeUtils::IsRealType(inputType1))
			{
				return {TypeUtils::WidestType(inputType0, inputType1)};
			}
			else if (TypeUtils::IsTypesEqual(inputType0, inputType1) && TypeUtils::IsType<BasicType>(inputType0))
			{
				return {inputType0};
			}
			else if (TypeUtils::IsType<ListType>(inputType0) || TypeUtils::IsType<ListType>(inputType1))
			{
				// Append the cells, finding the best element type

				const auto listType0 = TypeUtils::GetType<ListType>(inputType0);
				const auto listType1 = TypeUtils::GetType<ListType>(inputType1);

				const auto& elementTypes0 = (listType0 == nullptr) ? std::vector<Type *>({inputType0}) : listType0->GetElementTypes();
				const auto& elementTypes1 = (listType1 == nullptr) ? std::vector<Type *>({inputType1}) : listType1->GetElementTypes();

				if (!TypeUtils::IsSingleType(elementTypes0) && !TypeUtils::IsSingleType(elementTypes1))
				{
					std::vector<Type *> elementTypes = elementTypes0;
					elementTypes.insert(std::begin(elementTypes), std::begin(elementTypes1), std::end(elementTypes1));
					return {new ListType(elementTypes)};
				}
				else if (TypeUtils::IsSingleType(elementTypes0) && TypeUtils::IsTypesEqual(elementTypes0, elementTypes1))
				{
					return {new ListType(elementTypes0)};
				}
				return {new ListType(new WildcardType())};
			}
			else if (TypeUtils::IsType<EnumerationType>(inputType0))
			{
				// Append adds values to the enumeration

				const auto enumElementType = TypeUtils::GetType<EnumerationType>(inputType0)->GetElementType();
				Require(TypeUtils::IsTypesEqual(enumElementType, inputType1));
				return {inputType0};
			}
			break;
		}
		case BuiltinFunction::Primitive::Replicate:
		{
			const auto inputType0 = argumentTypes.at(0);
			const auto inputType1 = argumentTypes.at(1);
			Require(TypeUtils::IsIntegerType(inputType0) && (TypeUtils::IsType<BasicType>(inputType1) || TypeUtils::IsType<ListType>(inputType1)));

			if (const auto listType = TypeUtils::GetType<ListType>(inputType1))
			{
				if (TypeUtils::IsSingleType(listType->GetElementTypes()))
				{
					return {inputType1};
				}
				return {new ListType(new WildcardType())};
			}
			return {new ListType({inputType1})};
		}
		case BuiltinFunction::Primitive::Like:
		{
			const auto inputType0 = argumentTypes.at(0);
			const auto inputType1 = argumentTypes.at(1);
			Require(TypeUtils::IsCharacterType(inputType0) && TypeUtils::IsCharacterType(inputType1));
			return {new BasicType(BasicType::BasicKind::Boolean)};
		}
		case BuiltinFunction::Primitive::Compress:
		{
			const auto predicateType = argumentTypes.at(0);
			const auto dataType = argumentTypes.at(1);
			Require(TypeUtils::IsBooleanType(predicateType) && TypeUtils::IsType<BasicType>(dataType));
			return {dataType};
		}
		case BuiltinFunction::Primitive::Random_k:
		{
			const auto inputType0 = argumentTypes.at(0);
			const auto inputType1 = argumentTypes.at(1);
			Require(TypeUtils::IsIntegerType(inputType0) && TypeUtils::IsIntegerType(inputType1));
			return {new BasicType(BasicType::BasicKind::Int64)};
		}
		case BuiltinFunction::Primitive::IndexOf:
		{
			const auto inputType0 = argumentTypes.at(0);
			const auto inputType1 = argumentTypes.at(1);
			Require(
				(TypeUtils::IsRealType(inputType0) && TypeUtils::IsRealType(inputType1)) ||
				(TypeUtils::IsStringType(inputType0) && TypeUtils::IsStringType(inputType1)) ||
				(TypeUtils::IsSymbolType(inputType0) && TypeUtils::IsSymbolType(inputType1))
			);
			return {new BasicType(BasicType::BasicKind::Int64)};
		}
		case BuiltinFunction::Primitive::Take:
		case BuiltinFunction::Primitive::Drop:
		{
			const auto inputType0 = argumentTypes.at(0);
			const auto inputType1 = argumentTypes.at(1);
			Require(TypeUtils::IsIntegerType(inputType0) && TypeUtils::IsType<BasicType>(inputType1));
			return {inputType1};
		}
		case BuiltinFunction::Primitive::Order:
		{
			const auto inputType0 = argumentTypes.at(0);
			const auto inputType1 = argumentTypes.at(1);
			Require(TypeUtils::IsOrderableType(inputType0) || TypeUtils::IsType<ListType>(inputType0));
			Require(TypeUtils::IsBooleanType(inputType1));

			if (const auto listType = TypeUtils::GetType<ListType>(inputType0))
			{
				// List elements need to be comparable. Same return type as vector alternative

				Require(TypeUtils::ForallElements(listType, TypeUtils::IsOrderableType));
			}
			return {new BasicType(BasicType::BasicKind::Int64)};
		}
		case BuiltinFunction::Primitive::Member:
		{
			const auto inputType0 = argumentTypes.at(0);
			const auto inputType1 = argumentTypes.at(1);
			Require(
				(TypeUtils::IsRealType(inputType0) && TypeUtils::IsRealType(inputType1)) ||
				(TypeUtils::IsCharacterType(inputType0) && TypeUtils::IsCharacterType(inputType1)) ||
				(TypeUtils::IsTypesEqual(inputType0, inputType1) && TypeUtils::IsType<BasicType>(inputType0)) 
			);
			return {new BasicType(BasicType::BasicKind::Boolean)};
		}
		case BuiltinFunction::Primitive::Vector:
		{
			const auto inputType0 = argumentTypes.at(0);
			const auto inputType1 = argumentTypes.at(1);
			Require(TypeUtils::IsIntegerType(inputType0) && TypeUtils::IsType<BasicType>(inputType1));
			return {inputType1};
		}

		// Reduction
		case BuiltinFunction::Primitive::Length:
		{
			return {new BasicType(BasicType::BasicKind::Int64)};
		}
		case BuiltinFunction::Primitive::Sum:
		{
			const auto inputType = argumentTypes.at(0);
			Require(TypeUtils::IsRealType(inputType));
			if (TypeUtils::IsFloatType(inputType))
			{
				return {inputType};
			}
			if (TypeUtils::IsBasicType(inputType, BasicType::BasicKind::Int64))
			{
				return {new BasicType(BasicType::BasicKind::Int64)};
			}
			return {new BasicType(BasicType::BasicKind::Int32)};
		}
		case BuiltinFunction::Primitive::Average:
		{
			const auto inputType = argumentTypes.at(0);
			Require(TypeUtils::IsRealType(inputType));
			if (TypeUtils::IsExtendedType(inputType))
			{
				return {new BasicType(BasicType::BasicKind::Float64)};
			}
			return {new BasicType(BasicType::BasicKind::Float32)};
		}
		case BuiltinFunction::Primitive::Minimum:
		case BuiltinFunction::Primitive::Maximum:
		{
			const auto inputType = argumentTypes.at(0);
			Require(TypeUtils::IsRealType(inputType));
			return {inputType};
		}

		// List
		case BuiltinFunction::Primitive::Raze:
		{
			const auto inputType = argumentTypes.at(0);
			Require(TypeUtils::IsType<ListType>(inputType));

			// All element types must be the same, and non-wildcard

			const auto listType = TypeUtils::GetType<ListType>(inputType);
			if (const auto elementType = TypeUtils::GetReducedType(listType->GetElementTypes()))
			{
				Require(TypeUtils::IsType<BasicType>(elementType));
				return {elementType};
			}
			break;
		}
		case BuiltinFunction::Primitive::List:
		{
			// If the element type is set, then all cells have the same type, otherwise we can form a tuple

			if (const auto elementType = TypeUtils::GetReducedType(argumentTypes))
			{
				return {new ListType(elementType)};
			}
			return {new ListType(argumentTypes)};
		}
		case BuiltinFunction::Primitive::ToList:
		{
			const auto inputType = argumentTypes.at(0);
			Require(TypeUtils::IsType<BasicType>(inputType));
			return {new ListType(inputType)};
		}
		case BuiltinFunction::Primitive::Each:
		{
			const auto inputType0 = argumentTypes.at(0);
			const auto inputType1 = argumentTypes.at(1);
			Require(TypeUtils::IsType<FunctionType>(inputType0) && TypeUtils::IsType<ListType>(inputType1));

			// Evaluate the function on each cell

			const auto functionType = TypeUtils::GetType<FunctionType>(inputType0);
			const auto listType = TypeUtils::GetType<ListType>(inputType1);

			std::vector<Type *> returnTypes;
			for (const auto elementType : listType->GetElementTypes())
			{
				const auto types = AnalyzeCall(functionType, {elementType});
				Require(TypeUtils::IsSingleType(types));
				returnTypes.push_back(TypeUtils::GetSingleType(types));
			}
			return {new ListType(returnTypes)};
		}
		case BuiltinFunction::Primitive::EachItem:
		{
			const auto inputType0 = argumentTypes.at(0);
			const auto inputType1 = argumentTypes.at(1);
			const auto inputType2 = argumentTypes.at(2);
			Require(TypeUtils::IsType<FunctionType>(inputType0));
			Require(TypeUtils::IsType<ListType>(inputType1));
			Require(TypeUtils::IsType<ListType>(inputType2));

			const auto functionType = TypeUtils::GetType<FunctionType>(inputType0);
			const auto listType1 = TypeUtils::GetType<ListType>(inputType1);
			const auto listType2 = TypeUtils::GetType<ListType>(inputType2);

			const auto& elementTypes1 = listType1->GetElementTypes();
			const auto& elementTypes2 = listType2->GetElementTypes();

			auto elementCount1 = elementTypes1.size();
			auto elementCount2 = elementTypes2.size();
			Require(elementCount1 == elementCount2 || elementCount1 == 1 || elementCount2 == 1);

			auto count = std::max(elementCount1, elementCount2);
			std::vector<Type *> returnTypes;
			for (auto i = 0u; i < count; ++i)
			{
				// Get the arguments from the lists

				const auto l_inputType1 = elementTypes1.at((elementCount1 == 1) ? 0 : i);
				const auto l_inputType2 = elementTypes2.at((elementCount2 == 1) ? 0 : i);

				const auto types = AnalyzeCall(functionType, {l_inputType1, l_inputType2});
				Require(TypeUtils::IsSingleType(types));
				returnTypes.push_back(TypeUtils::GetSingleType(types));
			}
			return {new ListType(returnTypes)};
		}
		case BuiltinFunction::Primitive::EachLeft:
		{
			const auto inputType0 = argumentTypes.at(0);
			const auto inputType1 = argumentTypes.at(1);
			const auto inputType2 = argumentTypes.at(2);
			Require(TypeUtils::IsType<FunctionType>(inputType0));
			Require(TypeUtils::IsType<ListType>(inputType1));

			const auto functionType = TypeUtils::GetType<FunctionType>(inputType0);
			const auto listType1 = TypeUtils::GetType<ListType>(inputType1);

			std::vector<Type *> returnTypes;
			for (const auto elementType1 : listType1->GetElementTypes())
			{
				const auto types = AnalyzeCall(functionType, {elementType1, inputType2});
				Require(TypeUtils::IsSingleType(types));
				returnTypes.push_back(TypeUtils::GetSingleType(types));
			}
			return {new ListType(returnTypes)};
		}
		case BuiltinFunction::Primitive::EachRight:
		{
			const auto inputType0 = argumentTypes.at(0);
			const auto inputType1 = argumentTypes.at(1);
			const auto inputType2 = argumentTypes.at(2);
			Require(TypeUtils::IsType<FunctionType>(inputType0) && TypeUtils::IsType<ListType>(inputType2));

			const auto functionType = TypeUtils::GetType<FunctionType>(inputType0);
			const auto listType2 = TypeUtils::GetType<ListType>(inputType2);

			std::vector<Type *> returnTypes;
			for (const auto elementType2 : listType2->GetElementTypes())
			{
				const auto types = AnalyzeCall(functionType, {inputType1, elementType2});
				Require(TypeUtils::IsSingleType(types));
				returnTypes.push_back(TypeUtils::GetSingleType(types));
			}
			return {new ListType(returnTypes)};
		}
		case BuiltinFunction::Primitive::Match:
		{
			const auto inputType0 = argumentTypes.at(0);
			const auto inputType1 = argumentTypes.at(1);
			Require(TypeUtils::IsTypesEqual(inputType0, inputType1) || (TypeUtils::IsRealType(inputType0) && TypeUtils::IsRealType(inputType1)));
			return {new BasicType(BasicType::BasicKind::Boolean)}; // Scalar
		}
		
		// Date
		case BuiltinFunction::Primitive::Date:
		{
			const auto inputType = argumentTypes.at(0);
			Require(TypeUtils::IsDatetimeType(inputType));
			return {new BasicType(BasicType::BasicKind::Date)};
		}
		case BuiltinFunction::Primitive::DateYear:
		case BuiltinFunction::Primitive::DateMonth:
		{
			const auto inputType = argumentTypes.at(0);
			Require(TypeUtils::IsDatetimeType(inputType) || TypeUtils::IsDateType(inputType) || TypeUtils::IsMonthType(inputType));
			return {new BasicType(BasicType::BasicKind::Int16)};
		}
		case BuiltinFunction::Primitive::DateDay:
		{
			const auto inputType = argumentTypes.at(0);
			Require(TypeUtils::IsDatetimeType(inputType) || TypeUtils::IsDateType(inputType));
			return {new BasicType(BasicType::BasicKind::Int16)};
		}
		case BuiltinFunction::Primitive::Time:
		{
			const auto inputType = argumentTypes.at(0);
			Require(TypeUtils::IsDatetimeType(inputType));
			return {new BasicType(BasicType::BasicKind::Time)};
		}
		case BuiltinFunction::Primitive::TimeHour:
		case BuiltinFunction::Primitive::TimeMinute:
		{
			const auto inputType = argumentTypes.at(0);
			Require(TypeUtils::IsDatetimeType(inputType) || TypeUtils::IsMinuteType(inputType) || TypeUtils::IsSecondType(inputType) || TypeUtils::IsTimeType(inputType));
			return {new BasicType(BasicType::BasicKind::Int16)};
		}
		case BuiltinFunction::Primitive::TimeSecond:
		{
			const auto inputType = argumentTypes.at(0);
			Require(TypeUtils::IsDatetimeType(inputType) || TypeUtils::IsSecondType(inputType) || TypeUtils::IsTimeType(inputType));
			return {new BasicType(BasicType::BasicKind::Int16)};
		}
		case BuiltinFunction::Primitive::TimeMillisecond:
		{
			const auto inputType = argumentTypes.at(0);
			Require(TypeUtils::IsDatetimeType(inputType) || TypeUtils::IsTimeType(inputType));
			return {new BasicType(BasicType::BasicKind::Int16)};
		}
		case BuiltinFunction::Primitive::DatetimeAdd:
		case BuiltinFunction::Primitive::DatetimeSubtract:
		{
			const auto inputType0 = argumentTypes.at(0);
			const auto inputType1 = argumentTypes.at(1);
			const auto inputType2 = argumentTypes.at(2);
			Require(TypeUtils::IsCalendarType(inputType0) && TypeUtils::IsIntegerType(inputType1) && TypeUtils::IsSymbolType(inputType2));
			return {inputType0};
		}
		case BuiltinFunction::Primitive::DatetimeDifference:
		{
			const auto inputType0 = argumentTypes.at(0);
			const auto inputType1 = argumentTypes.at(1);
			Require(TypeUtils::IsTypesEqual(inputType0, inputType1) && TypeUtils::IsCalendarType(inputType0));
			if (TypeUtils::IsDatetimeType(inputType0))
			{
				return {new BasicType(BasicType::BasicKind::Int64)};
			}
			return {new BasicType(BasicType::BasicKind::Int32)};
		}

		// Database
		case BuiltinFunction::Primitive::Enum:
		{
			const auto inputType0 = argumentTypes.at(0);
			const auto inputType1 = argumentTypes.at(1);
			Require(TypeUtils::IsTypesEqual(inputType0, inputType1) && (TypeUtils::IsType<BasicType>(inputType0) || TypeUtils::IsType<ListType>(inputType0)));

			if (const auto listType = TypeUtils::GetType<ListType>(inputType0))
			{
				const auto elementType = TypeUtils::GetReducedType(listType->GetElementTypes());
				Require(elementType != nullptr && TypeUtils::IsType<BasicType>(elementType));
			}
			return {new EnumerationType(inputType0)};
		}
		case BuiltinFunction::Primitive::Dictionary:
		{
			const auto inputType0 = argumentTypes.at(0);
			const auto inputType1 = argumentTypes.at(1);
			return {new DictionaryType(inputType0, inputType1)};
		}
		case BuiltinFunction::Primitive::Table:
		{
			const auto inputType0 = argumentTypes.at(0);
			const auto inputType1 = argumentTypes.at(1);
			Require(TypeUtils::IsSymbolType(inputType0) && TypeUtils::IsType<ListType>(inputType1));

			// All list elements must be vectors

			auto listType = TypeUtils::GetType<ListType>(inputType1);
			Require(TypeUtils::ForallElements(listType, TypeUtils::IsType<BasicType>));
			return {new TableType()};
		}
		case BuiltinFunction::Primitive::KeyedTable:
		{
			const auto inputType0 = argumentTypes.at(0);
			const auto inputType1 = argumentTypes.at(1);
			Require(TypeUtils::IsType<TableType>(inputType0) && TypeUtils::IsType<TableType>(inputType1));
			return {new KeyedTableType()};
		}
		case BuiltinFunction::Primitive::Keys:
		{
			const auto inputType = argumentTypes.at(0);
			if (TypeUtils::IsType<DictionaryType>(inputType))
			{
				const auto dictionaryType = TypeUtils::GetType<DictionaryType>(inputType);
				const auto keyType = dictionaryType->GetKeyType();
				if (TypeUtils::IsType<BasicType>(keyType) || TypeUtils::IsType<ListType>(keyType))
				{
					return {keyType};
				}
				return {new ListType(keyType)};
			}
			else if (TypeUtils::IsType<TableType>(inputType))
			{
				return {TypeUtils::GetType<TableType>(inputType)->GetKeyType()};
			}
			else if (TypeUtils::IsType<EnumerationType>(inputType))
			{
				return {TypeUtils::GetType<EnumerationType>(inputType)->GetElementType()};
			}
			else if (TypeUtils::IsType<KeyedTableType>(inputType))
			{
				return {TypeUtils::GetType<KeyedTableType>(inputType)->GetKeyType()};
			}
			break;
		}
		case BuiltinFunction::Primitive::Values:
		{
			const auto inputType = argumentTypes.at(0);
			if (TypeUtils::IsType<DictionaryType>(inputType))
			{
				const auto dictionaryType = TypeUtils::GetType<DictionaryType>(inputType);
				const auto valueType = dictionaryType->GetValueType();
				if (TypeUtils::IsType<ListType>(valueType))
				{
					return {valueType};
				}
				return {new ListType(valueType)};
			}
			else if (TypeUtils::IsType<TableType>(inputType))
			{
				return {TypeUtils::GetType<TableType>(inputType)->GetValueType()};
			}
			else if (TypeUtils::IsType<EnumerationType>(inputType))
			{
				return {new BasicType(BasicType::BasicKind::Int64)};
			}
			else if (TypeUtils::IsType<KeyedTableType>(inputType))
			{
				return {TypeUtils::GetType<KeyedTableType>(inputType)->GetValueType()};
			}
			break;
		}
		case BuiltinFunction::Primitive::Meta:
		{
			const auto inputType = argumentTypes.at(0);
			Require(TypeUtils::IsType<TableType>(inputType) || TypeUtils::IsType<KeyedTableType>(inputType));
			return {new TableType()};
		}
		case BuiltinFunction::Primitive::Fetch:
		{
			const auto inputType = argumentTypes.at(0);
			Require(TypeUtils::IsType<EnumerationType>(inputType));
			return {TypeUtils::GetType<EnumerationType>(inputType)->GetElementType()};
		}
		case BuiltinFunction::Primitive::ColumnValue:
		{
			const auto inputType0 = argumentTypes.at(0);
			const auto inputType1 = argumentTypes.at(1);
			Require(TypeUtils::IsType<TableType>(inputType0) && TypeUtils::IsSymbolType(inputType1));

			// A column value call is intentionally untyped since it comes from the runtime system.
			// It must be cast before assigning to a variable

			return {new HorseIR::WildcardType()};
		}
		case BuiltinFunction::Primitive::LoadTable:
		{
			const auto inputType = argumentTypes.at(0);
			Require(TypeUtils::IsSymbolType(inputType));
			return {new TableType()};
		}
		case BuiltinFunction::Primitive::JoinIndex:
		{
			Require(AnalyzeJoinArguments(argumentTypes));
			return {new ListType(new BasicType(BasicType::BasicKind::Int64))};
		}

		// Indexing
		case BuiltinFunction::Primitive::Index:
		{
			const auto inputType0 = argumentTypes.at(0);
			const auto inputType1 = argumentTypes.at(1);
			Require(TypeUtils::IsIntegerType(inputType1));

			if (TypeUtils::IsType<BasicType>(inputType0))
			{
				return {inputType0};
			}
			else if (const auto listType0 = TypeUtils::GetType<ListType>(inputType0))
			{
				return {TypeUtils::GetReducedType(listType0->GetElementTypes())};
			}
			break;
		}
		case BuiltinFunction::Primitive::IndexAssignment:
		{
			const auto inputType0 = argumentTypes.at(0);
			const auto inputType1 = argumentTypes.at(1);
			const auto inputType2 = argumentTypes.at(2);
			Require(TypeUtils::IsType<BasicType>(inputType0) && TypeUtils::IsIntegerType(inputType1) && TypeUtils::IsAssignableType(inputType0, inputType2));
			return {inputType0};
		}

		// Other
		case BuiltinFunction::Primitive::LoadCSV:
		{
			return {new TableType()};
		}
		case BuiltinFunction::Primitive::Print:
		{
			return {new BasicType(BasicType::BasicKind::Int64)};
		}
		case BuiltinFunction::Primitive::String:
		{
			return {new BasicType(BasicType::BasicKind::String)};
		}
		case BuiltinFunction::Primitive::SubString:
		{
			const auto inputType0 = argumentTypes.at(0);
			const auto inputType1 = argumentTypes.at(1);
			Require(TypeUtils::IsCharacterType(inputType0));
			Require(TypeUtils::IsIntegerType(inputType1));
			return {inputType0};
		}

		// GPU
		case BuiltinFunction::Primitive::GPUOrderLib:
		{
			// @GPU.order_lib(@init, @sort, [@sort_shared], data, [order])

			Require(argumentTypes.size() >= 3);
			const auto isShared = (TypeUtils::IsType<FunctionType>(argumentTypes.at(2)));
			Require(argumentTypes.size() == (3 + isShared) || argumentTypes.size() == (4 + isShared));

			const auto initType = argumentTypes.at(0);
			const auto sortType = argumentTypes.at(1);
			const auto dataType = argumentTypes.at(2 + isShared);

			Require(TypeUtils::IsType<FunctionType>(initType));
			Require(TypeUtils::IsType<FunctionType>(sortType));

			auto initFunction = TypeUtils::GetType<FunctionType>(initType);
			auto sortFunction = TypeUtils::GetType<FunctionType>(sortType);

			Require(TypeUtils::IsOrderableType(dataType) || TypeUtils::IsType<ListType>(dataType));

			if (const auto listType = TypeUtils::GetType<ListType>(dataType))
			{
				// List elements need to be comparable. Same return type as vector alternative

				Require(TypeUtils::ForallElements(listType, TypeUtils::IsOrderableType));
			}

			if (argumentTypes.size() == (3 + isShared))
			{
				// Init sort call

				const auto initCallTypes = AnalyzeCall(initFunction, {dataType});
				Require(initCallTypes.size() == 2);
				Require(TypeUtils::IsBasicType(initCallTypes.at(0), BasicType::BasicKind::Int64));
				Require(TypeUtils::IsTypesEqual(initCallTypes.at(1), dataType));

				// Sort call

				const auto sortCallTypes = AnalyzeCall(sortFunction, {initCallTypes.at(0), dataType});
				Require(TypeUtils::IsEmptyType(sortCallTypes));

				if (isShared)
				{
					auto sharedFunction = TypeUtils::GetType<FunctionType>(argumentTypes.at(2));
					const auto sharedCallTypes = AnalyzeCall(sharedFunction, {initCallTypes.at(0), dataType});
					Require(TypeUtils::IsEmptyType(sharedCallTypes));
				}

				return {initCallTypes.at(0)};
			}

			const auto orderType = argumentTypes.at(3 + isShared);
			Require(TypeUtils::IsBooleanType(orderType));

			// Init sort call

			const auto initCallTypes = AnalyzeCall(initFunction, {dataType, orderType});
			Require(initCallTypes.size() == 2);
			Require(TypeUtils::IsBasicType(initCallTypes.at(0), BasicType::BasicKind::Int64));
			Require(TypeUtils::IsTypesEqual(initCallTypes.at(1), dataType));

			// Sort call

			const auto sortCallTypes = AnalyzeCall(sortFunction, {initCallTypes.at(0), dataType, orderType});
			Require(TypeUtils::IsEmptyType(sortCallTypes));

			if (isShared)
			{
				auto sharedFunction = TypeUtils::GetType<FunctionType>(argumentTypes.at(2));
				const auto sharedCallTypes = AnalyzeCall(sharedFunction, {initCallTypes.at(0), dataType, orderType});
				Require(TypeUtils::IsEmptyType(sharedCallTypes));
			}

			return {initCallTypes.at(0)};
		}
		case BuiltinFunction::Primitive::GPUOrderInit:
		{
			const auto inputType0 = argumentTypes.at(0);
			const auto inputType1 = argumentTypes.at(1);

			Require(TypeUtils::IsOrderableType(inputType0) || TypeUtils::IsType<ListType>(inputType0));

			if (const auto listType = TypeUtils::GetType<ListType>(inputType0))
			{
				// List elements need to be comparable

				Require(TypeUtils::ForallElements(listType, TypeUtils::IsOrderableType));
			}

			Require(TypeUtils::IsBooleanType(inputType1));

			return {new BasicType(BasicType::BasicKind::Int64), inputType0};
		}
		case BuiltinFunction::Primitive::GPUOrder:
		case BuiltinFunction::Primitive::GPUOrderShared:
		{
			const auto inputType0 = argumentTypes.at(0);
			const auto inputType1 = argumentTypes.at(1);
			const auto inputType2 = argumentTypes.at(2);

			Require(TypeUtils::IsBasicType(inputType0, BasicType::BasicKind::Int64));
			Require(TypeUtils::IsOrderableType(inputType1) || TypeUtils::IsType<ListType>(inputType1));

			if (const auto listType = TypeUtils::GetType<ListType>(inputType1))
			{
				// List elements need to be comparable

				Require(TypeUtils::ForallElements(listType, TypeUtils::IsOrderableType));
			}
			Require(TypeUtils::IsBooleanType(inputType2));

			// GPU sort is in place
			return {};
		}
		case BuiltinFunction::Primitive::GPUGroupLib:
		{
			// @GPU.group_lib(@init, @sort, [@sort_shared], @group, data)

			const auto isShared = (argumentTypes.size() == 5);
			Require(argumentTypes.size() == (isShared + 4));

			const auto initType  = argumentTypes.at(0);
			const auto sortType  = argumentTypes.at(1);
			const auto groupType = argumentTypes.at(2 + isShared);
			const auto dataType  = argumentTypes.at(3 + isShared);

			Require(TypeUtils::IsType<FunctionType>(initType));
			Require(TypeUtils::IsType<FunctionType>(sortType));
			Require(TypeUtils::IsType<FunctionType>(groupType));

			auto initFunction = TypeUtils::GetType<FunctionType>(initType);
			auto sortFunction = TypeUtils::GetType<FunctionType>(sortType);
			auto groupFunction = TypeUtils::GetType<FunctionType>(groupType);

			Require(TypeUtils::IsOrderableType(dataType) || TypeUtils::IsType<ListType>(dataType));

			if (const auto listType = TypeUtils::GetType<ListType>(dataType))
			{
				// List elements need to be comparable. Same return type as vector alternative

				Require(TypeUtils::ForallElements(listType, TypeUtils::IsOrderableType));
			}

			// Init sort call

			const auto initCallTypes = AnalyzeCall(initFunction, {dataType});
			Require(initCallTypes.size() == 2);
			Require(TypeUtils::IsBasicType(initCallTypes.at(0), BasicType::BasicKind::Int64));
			Require(TypeUtils::IsTypesEqual(initCallTypes.at(1), dataType));

			// Sort call

			const auto sortCallTypes = AnalyzeCall(sortFunction, {initCallTypes.at(0), dataType});
			Require(TypeUtils::IsEmptyType(sortCallTypes));

			if (isShared)
			{
				// Sort shared call

				const auto sharedType = argumentTypes.at(2);
				Require(TypeUtils::IsType<FunctionType>(sharedType));
				auto sharedFunction = TypeUtils::GetType<FunctionType>(sharedType);

				const auto sharedCallTypes = AnalyzeCall(sharedFunction, {initCallTypes.at(0), dataType});
				Require(TypeUtils::IsEmptyType(sharedCallTypes));
			}

			// Group call

			const auto groupCallTypes = AnalyzeCall(groupFunction, {initCallTypes.at(0), dataType});
			Require(groupCallTypes.size() == 2);
			Require(TypeUtils::IsBasicType(groupCallTypes.at(0), BasicType::BasicKind::Int64));
			Require(TypeUtils::IsBasicType(groupCallTypes.at(1), BasicType::BasicKind::Int64));

			return {new DictionaryType(new BasicType(BasicType::BasicKind::Int64), new BasicType(BasicType::BasicKind::Int64))};
		}
		case BuiltinFunction::Primitive::GPUGroup:
		{
			const auto inputType0 = argumentTypes.at(0);
			const auto inputType1 = argumentTypes.at(1);

			Require(TypeUtils::IsBasicType(inputType0, BasicType::BasicKind::Int64));
			Require(TypeUtils::IsOrderableType(inputType1) || TypeUtils::IsType<ListType>(inputType1));

			if (const auto listType = TypeUtils::GetType<ListType>(inputType1))
			{
				// List elements need to be comparable. Same return type as vector alternative

				Require(TypeUtils::ForallElements(listType, TypeUtils::IsOrderableType));
			}

			return {new BasicType(BasicType::BasicKind::Int64), new BasicType(BasicType::BasicKind::Int64)};
		}
		case BuiltinFunction::Primitive::GPUUniqueLib:
		{
			// @GPU.unique_lib(@init, @sort, [@sort_shared], @unique, data)

			const auto isShared = (argumentTypes.size() == 5);
			Require(argumentTypes.size() == (isShared + 4));

			const auto initType = argumentTypes.at(0);
			const auto sortType = argumentTypes.at(1);
			const auto uniqueType = argumentTypes.at(2 + isShared);
			const auto dataType   = argumentTypes.at(3 + isShared);

			Require(TypeUtils::IsType<FunctionType>(initType));
			Require(TypeUtils::IsType<FunctionType>(sortType));
			Require(TypeUtils::IsType<FunctionType>(uniqueType));

			auto initFunction = TypeUtils::GetType<FunctionType>(initType);
			auto sortFunction = TypeUtils::GetType<FunctionType>(sortType);
			auto uniqueFunction = TypeUtils::GetType<FunctionType>(uniqueType);

			Require(TypeUtils::IsOrderableType(dataType));

			// Init sort call

			const auto initCallTypes = AnalyzeCall(initFunction, {dataType});
			Require(initCallTypes.size() == 2);
			Require(TypeUtils::IsBasicType(initCallTypes.at(0), BasicType::BasicKind::Int64));
			Require(TypeUtils::IsTypesEqual(initCallTypes.at(1), dataType));

			// Sort call

			const auto sortCallTypes = AnalyzeCall(sortFunction, {initCallTypes.at(0), dataType});
			Require(TypeUtils::IsEmptyType(sortCallTypes));

			if (isShared)
			{
				// Sort shared call

				const auto sharedType = argumentTypes.at(2);
				Require(TypeUtils::IsType<FunctionType>(sharedType));
				auto sharedFunction = TypeUtils::GetType<FunctionType>(sharedType);

				const auto sharedCallTypes = AnalyzeCall(sharedFunction, {initCallTypes.at(0), dataType});
				Require(TypeUtils::IsEmptyType(sharedCallTypes));
			}

			// Unique call

			const auto uniqueCallTypes = AnalyzeCall(uniqueFunction, {initCallTypes.at(0), dataType});
			Require(uniqueCallTypes.size() == 1);
			Require(TypeUtils::IsBasicType(uniqueCallTypes.at(0), BasicType::BasicKind::Int64));

			return {uniqueCallTypes.at(0)};
		}
		case BuiltinFunction::Primitive::GPUUnique:
		{
			const auto inputType0 = argumentTypes.at(0);
			const auto inputType1 = argumentTypes.at(1);

			Require(TypeUtils::IsBasicType(inputType0, BasicType::BasicKind::Int64));
			Require(TypeUtils::IsOrderableType(inputType1));

			return {new BasicType(BasicType::BasicKind::Int64)};
		}
		case BuiltinFunction::Primitive::GPULoopJoinLib:
		{
			// @GPU.loop_join_lib(@count, @join, left, right)

			const auto inputType0 = argumentTypes.at(0);
			const auto inputType1 = argumentTypes.at(1);
			const auto inputType2 = argumentTypes.at(2);
			const auto inputType3 = argumentTypes.at(3);

			Require(TypeUtils::IsType<FunctionType>(inputType0));
			Require(TypeUtils::IsType<FunctionType>(inputType1));

			auto functionType0 = TypeUtils::GetType<FunctionType>(inputType0);
			auto functionType1 = TypeUtils::GetType<FunctionType>(inputType1);

			Require(TypeUtils::IsType<BasicType>(inputType2) || TypeUtils::IsType<ListType>(inputType2));
			Require(TypeUtils::IsType<BasicType>(inputType3) || TypeUtils::IsType<ListType>(inputType3));

			if (const auto listType2 = TypeUtils::GetType<ListType>(inputType2))
			{
				Require(TypeUtils::ForallElements(listType2, TypeUtils::IsType<BasicType>));
			}
			if (const auto listType3 = TypeUtils::GetType<ListType>(inputType3))
			{
				Require(TypeUtils::ForallElements(listType3, TypeUtils::IsType<BasicType>));
			}

			// Count call

			const auto callTypes0 = AnalyzeCall(functionType0, {inputType2, inputType3});
			Require(callTypes0.size() == 2);
			Require(TypeUtils::IsBasicType(callTypes0.at(0), BasicType::BasicKind::Int64));
			Require(TypeUtils::IsBasicType(callTypes0.at(1), BasicType::BasicKind::Int64));

			// Join call

			const auto callTypes1 = AnalyzeCall(functionType1, {inputType2, inputType3, callTypes0.at(0), callTypes0.at(1)});
			Require(TypeUtils::IsType<ListType>(callTypes1.at(0)));

			const auto listCallType1 = TypeUtils::GetType<ListType>(callTypes1.at(0));
			const auto elementCallType1 = TypeUtils::GetReducedType(listCallType1->GetElementTypes());
			Require(TypeUtils::IsBasicType(elementCallType1, BasicType::BasicKind::Int64));

			return {callTypes1.at(0)};
		}
		case BuiltinFunction::Primitive::GPULoopJoinCount:
		{
			Require(AnalyzeJoinArguments(argumentTypes));
			return {new BasicType(BasicType::BasicKind::Int64), new BasicType(BasicType::BasicKind::Int64)};
		}
		case BuiltinFunction::Primitive::GPULoopJoin:
		{
			std::vector<Type *> joinTypes(std::begin(argumentTypes), std::end(argumentTypes) - 2);
			Require(AnalyzeJoinArguments(joinTypes));

			const auto offsetsType = argumentTypes.at(argumentTypes.size() - 2);
			const auto countType = argumentTypes.at(argumentTypes.size() - 1);
			Require(TypeUtils::IsBasicType(offsetsType, BasicType::BasicKind::Int64));
			Require(TypeUtils::IsBasicType(countType, BasicType::BasicKind::Int64));

			return {new ListType(new BasicType(BasicType::BasicKind::Int64))};
		}
		case BuiltinFunction::Primitive::GPUHashJoinLib:
		{
			// @GPU.hash_join_lib(@hash, @count, @join, left, right)

			const auto inputType0 = argumentTypes.at(0);
			const auto inputType1 = argumentTypes.at(1);
			const auto inputType2 = argumentTypes.at(2);
			const auto inputType3 = argumentTypes.at(3);
			const auto inputType4 = argumentTypes.at(4);

			Require(TypeUtils::IsType<FunctionType>(inputType0));
			Require(TypeUtils::IsType<FunctionType>(inputType1));
			Require(TypeUtils::IsType<FunctionType>(inputType2));

			auto functionType0 = TypeUtils::GetType<FunctionType>(inputType0);
			auto functionType1 = TypeUtils::GetType<FunctionType>(inputType1);
			auto functionType2 = TypeUtils::GetType<FunctionType>(inputType2);

			Require(TypeUtils::IsType<BasicType>(inputType3) || TypeUtils::IsType<ListType>(inputType3));
			Require(TypeUtils::IsType<BasicType>(inputType4) || TypeUtils::IsType<ListType>(inputType4));

			if (const auto listType3 = TypeUtils::GetType<ListType>(inputType3))
			{
				Require(TypeUtils::ForallElements(listType3, TypeUtils::IsType<BasicType>));
			}
			if (const auto listType4 = TypeUtils::GetType<ListType>(inputType4))
			{
				Require(TypeUtils::ForallElements(listType4, TypeUtils::IsType<BasicType>));
			}

			// Hash call

			const auto callTypes0 = AnalyzeCall(functionType0, {inputType3});
			Require(callTypes0.size() == 2);
			Require(TypeUtils::IsTypesEqual(callTypes0.at(0), inputType3));
			Require(TypeUtils::IsBasicType(callTypes0.at(1), BasicType::BasicKind::Int64));

			// Count call

			const auto callTypes1 = AnalyzeCall(functionType1, {callTypes0.at(0), inputType4});
			Require(callTypes1.size() == 2);
			Require(TypeUtils::IsBasicType(callTypes1.at(0), BasicType::BasicKind::Int64));
			Require(TypeUtils::IsBasicType(callTypes1.at(1), BasicType::BasicKind::Int64));

			// Join call

			const auto callTypes2 = AnalyzeCall(functionType2, {callTypes0.at(0), callTypes0.at(1), inputType4, callTypes1.at(0), callTypes1.at(1)});
			Require(TypeUtils::IsType<ListType>(callTypes2.at(0)));

			const auto listCallType2 = TypeUtils::GetType<ListType>(callTypes2.at(0));
			const auto elementCallType2 = TypeUtils::GetReducedType(listCallType2->GetElementTypes());
			Require(TypeUtils::IsBasicType(elementCallType2, BasicType::BasicKind::Int64));

			return {callTypes2.at(0)};
		}
		case BuiltinFunction::Primitive::GPUHashCreate:
		{
			const auto inputType = argumentTypes.at(0);

			Require(TypeUtils::IsType<BasicType>(inputType) || TypeUtils::IsType<ListType>(inputType));
			if (const auto listType = TypeUtils::GetType<ListType>(inputType))
			{
				Require(TypeUtils::ForallElements(listType, TypeUtils::IsType<BasicType>));
			}

			return {inputType, new BasicType(BasicType::BasicKind::Int64)};
		}
		case BuiltinFunction::Primitive::GPUHashJoinCount:
		{
			const auto hashKeyType = argumentTypes.at(0);
			const auto rightType = argumentTypes.at(1);

			Require(
				(TypeUtils::IsType<BasicType>(rightType) && TypeUtils::IsType<BasicType>(hashKeyType)) ||
				(TypeUtils::IsType<ListType>(rightType) && TypeUtils::IsType<ListType>(hashKeyType))
			);

			if (TypeUtils::IsType<ListType>(rightType))
			{
				const auto& elementTypes1 = TypeUtils::GetType<ListType>(rightType)->GetElementTypes();
				const auto& elementTypes2 = TypeUtils::GetType<ListType>(hashKeyType)->GetElementTypes();

				auto elementCount1 = elementTypes1.size();
				auto elementCount2 = elementTypes2.size();

				auto count = std::max({elementCount1, elementCount2});
				for (auto i = 0u; i < count; ++i)
				{
					// Get the arguments from the lists and the function

					const auto l_inputType1 = elementTypes1.at((elementCount1 == 1) ? 0 : i);
					const auto l_inputType2 = elementTypes2.at((elementCount2 == 1) ? 0 : i);

					Require(TypeUtils::IsComparableTypes(l_inputType1, l_inputType2));
				}
			}
			else
			{
				Require(TypeUtils::IsComparableTypes(rightType, hashKeyType));
			}
			return {new BasicType(BasicType::BasicKind::Int64), new BasicType(BasicType::BasicKind::Int64)};
		}
		case BuiltinFunction::Primitive::GPUHashJoin:
		{
			const auto hashKeyType = argumentTypes.at(0);
			const auto hashValueType = argumentTypes.at(1);
			const auto rightType = argumentTypes.at(2);
			const auto offsetsType = argumentTypes.at(3);
			const auto countType = argumentTypes.at(4);

			Require(TypeUtils::IsBasicType(hashValueType, BasicType::BasicKind::Int64));

			Require(
				(TypeUtils::IsType<BasicType>(rightType) && TypeUtils::IsType<BasicType>(hashKeyType)) ||
				(TypeUtils::IsType<ListType>(rightType) && TypeUtils::IsType<ListType>(hashKeyType))
			);

			if (TypeUtils::IsType<ListType>(rightType))
			{
				const auto& elementTypes1 = TypeUtils::GetType<ListType>(rightType)->GetElementTypes();
				const auto& elementTypes2 = TypeUtils::GetType<ListType>(hashKeyType)->GetElementTypes();

				auto elementCount1 = elementTypes1.size();
				auto elementCount2 = elementTypes2.size();

				auto count = std::max({elementCount1, elementCount2});
				for (auto i = 0u; i < count; ++i)
				{
					// Get the arguments from the lists and the function

					const auto l_inputType1 = elementTypes1.at((elementCount1 == 1) ? 0 : i);
					const auto l_inputType2 = elementTypes2.at((elementCount2 == 1) ? 0 : i);

					Require(TypeUtils::IsComparableTypes(l_inputType1, l_inputType2));
				}
			}
			else
			{
				Require(TypeUtils::IsComparableTypes(rightType, hashKeyType));
			}

			Require(TypeUtils::IsBasicType(offsetsType, BasicType::BasicKind::Int64));
			Require(TypeUtils::IsBasicType(countType, BasicType::BasicKind::Int64));

			return {new ListType(new BasicType(BasicType::BasicKind::Int64))};
		}
		default:
		{
			Utils::Logger::LogError("Type analysis does not support builtin function '" + function->GetName() + "'");
		}
	}
	
	// If we cannot infer a return type, report a type error

	TypeError(function, argumentTypes);
}

bool TypeChecker::AnalyzeJoinArguments(const std::vector<Type *>& argumentTypes) const
{
#define RequireJoin(x) if (!(x)) return false

	// At least one function must be present, in addition to 2 input variables

	RequireJoin(argumentTypes.size() >= 3);

	auto functionCount = argumentTypes.size() - 2;
	const auto inputType1 = argumentTypes.at(functionCount);
	const auto inputType2 = argumentTypes.at(functionCount + 1);
	RequireJoin(
		(TypeUtils::IsType<BasicType>(inputType1) && TypeUtils::IsType<BasicType>(inputType2)) ||
		(TypeUtils::IsType<ListType>(inputType1) && TypeUtils::IsType<ListType>(inputType2))
	);

	if (TypeUtils::IsType<ListType>(inputType1))
	{
		const auto& elementTypes1 = TypeUtils::GetType<ListType>(inputType1)->GetElementTypes();
		const auto& elementTypes2 = TypeUtils::GetType<ListType>(inputType2)->GetElementTypes();

		auto elementCount1 = elementTypes1.size();
		auto elementCount2 = elementTypes2.size();
		if (elementCount1 == elementCount2)
		{
			RequireJoin(functionCount == 1 || elementCount1 == 1 || elementCount1 == functionCount);
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
			const auto inputType = argumentTypes.at((functionCount == 1) ? 0 : i);
			RequireJoin(TypeUtils::IsType<FunctionType>(inputType));

			// Get the arguments from the lists and the function

			const auto functionType = TypeUtils::GetType<FunctionType>(inputType);
			const auto l_inputType1 = elementTypes1.at((elementCount1 == 1) ? 0 : i);
			const auto l_inputType2 = elementTypes2.at((elementCount2 == 1) ? 0 : i);

			const auto returnType = AnalyzeCall(functionType, {l_inputType1, l_inputType2});
			RequireJoin(TypeUtils::IsSingleType(returnType) && TypeUtils::IsBooleanType(TypeUtils::GetSingleType(returnType)));
		}
	}
	else
	{
		// If the inputs are vectors, require a single function

		RequireJoin(functionCount == 1);

		const auto inputType0 = argumentTypes.at(0);
		RequireJoin(TypeUtils::IsType<FunctionType>(inputType0));
		const auto functionType = TypeUtils::GetType<FunctionType>(inputType0);

		const auto returnType = AnalyzeCall(functionType, {inputType1, inputType2});
		RequireJoin(TypeUtils::IsSingleType(returnType) && TypeUtils::IsBooleanType(TypeUtils::GetSingleType(returnType)));
	}
	return true;
}

void TypeChecker::VisitOut(Identifier *identifier)
{
	// Get the type associated with the identifier

	const auto symbol = identifier->GetSymbol();
	switch (symbol->kind)
	{
		case SymbolTable::Symbol::Kind::Function:
		{
			const auto functionDeclaration = static_cast<const FunctionDeclaration *>(symbol->node);
			switch (functionDeclaration->GetKind())
			{
				case FunctionDeclaration::Kind::Builtin:
				{
					// Builtin functions have variable input/outputs and must be checked separately

					const auto function = static_cast<const BuiltinFunction *>(functionDeclaration);
					identifier->SetType(new FunctionType(function));
					break;
				}
				case FunctionDeclaration::Kind::Definition:
				{
					const auto function = static_cast<const Function *>(functionDeclaration);
					identifier->SetType(new FunctionType(function));
					break;
				}
				default:
					Utils::Logger::LogError("Unsupported function kind");
			}
			break;
		}
		case SymbolTable::Symbol::Kind::Variable:
			identifier->SetType(dynamic_cast<const VariableDeclaration *>(symbol->node)->GetType());
			break;
		case SymbolTable::Symbol::Kind::Module:
			Utils::Logger::LogError("Module '" + PrettyPrinter::PrettyString(identifier) + "' used as a variable or function");
	}
}

void TypeChecker::VisitOut(VectorLiteral *literal)
{
	// Propagate the value type to the literal

	literal->SetType(new BasicType(literal->GetBasicKind()));
}

void TypeChecker::VisitOut(FunctionLiteral *literal)
{
	// Propagate the function from the literal expression to the expression type

	literal->SetType(literal->GetIdentifier()->GetType());
}

void TypeChecker::VisitOut(EnumerationType *type)
{
	// Ensure the enum element type is legal, not enforced in the parser

	const auto elementType = type->GetElementType();
	if (TypeUtils::IsType<BasicType>(elementType))
	{
		return;
	}
	if (const auto listType = TypeUtils::GetType<ListType>(elementType))
	{
		const auto elementType = TypeUtils::GetReducedType(listType->GetElementTypes());
		if (elementType != nullptr && TypeUtils::IsType<BasicType>(elementType))
		{
			return;
		}
	}
	Utils::Logger::LogError("Invalid key/value types for enumeration '" + PrettyPrinter::PrettyString(type) + "'");
}

}
