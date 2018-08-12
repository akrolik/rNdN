#pragma once

#include "HorseIR/Tree/Types/Type.h"
#include "HorseIR/Tree/Types/BasicType.h"
#include "HorseIR/Tree/Types/DictionaryType.h"
#include "HorseIR/Tree/Types/FunctionType.h"
#include "HorseIR/Tree/Types/ListType.h"

#include "Utils/Logger.h"

namespace HorseIR {

static BasicType *GetBasicType(Type *type)
{
	if (type->GetKind() == Type::Kind::Basic)
	{
		return static_cast<BasicType *>(type);
	}
	return nullptr;
}

static const BasicType *GetBasicType(const Type *type)
{
	if (type->GetKind() == Type::Kind::Basic)
	{
		return static_cast<const BasicType *>(type);
	}
	return nullptr;
}

static const ListType *GetListType(const Type *type)
{
	if (type->GetKind() != Type::Kind::List)
	{
		return static_cast<const ListType *>(type);
	}
	return nullptr;
}

static const DictionaryType *GetDictionaryType(const Type *type)
{
	if (type->GetKind() != Type::Kind::Dictionary)
	{
		return static_cast<const DictionaryType *>(type);
	}
	return nullptr;
}

static const TableType *GetTableType(const Type *type)
{
	if (type->GetKind() != Type::Kind::Table)
	{
		return static_cast<const TableType *>(type);
	}
	return nullptr;
}

static FunctionType *GetFunctionType(Type *type)
{
	if (auto basicType = GetBasicType(type); type != nullptr && basicType->GetKind() == BasicType::Kind::Function)
	{
		return static_cast<FunctionType *>(type);
	}
	return nullptr;
}

static const FunctionType *GetFunctionType(const Type *type)
{
	if (auto basicType = GetBasicType(type); type != nullptr && basicType->GetKind() == BasicType::Kind::Function)
	{
		return static_cast<const FunctionType *>(type);
	}
	return nullptr;
}

static bool IsBasicType(const Type *type, BasicType::Kind kind)
{
	if (auto basicType = GetBasicType(type); basicType != nullptr)
	{
		return (basicType->GetKind() == kind);
	}
	return false;
}

static bool IsBasicType(const Type *type)
{
	return (type->GetKind() == Type::Kind::Basic);
}

static bool IsListType(const Type *type)
{
	return (type->GetKind() == Type::Kind::List);
}

static bool IsTableType(const Type *type)
{
	return (type->GetKind() == Type::Kind::Table);
}

static bool IsDictionaryType(const Type *type)
{
	return (type->GetKind() == Type::Kind::Dictionary);
}

static bool IsEnumerationType(const Type *type)
{
	return (type->GetKind() == Type::Kind::Enumeration);
}

static bool IsKeyedTableType(const Type *type)
{
	return (type->GetKind() == Type::Kind::KeyedTable);
}

static bool IsBoolType(const Type *type)
{                       
	return IsBasicType(type, BasicType::Kind::Bool);
}

static bool IsIntegerType(const Type *type)
{                       
	if (auto basicType = GetBasicType(type); basicType != nullptr)
	{
		switch (basicType->GetKind())
		{
			case BasicType::Kind::Bool:
			case BasicType::Kind::Int8:
			case BasicType::Kind::Int16:
			case BasicType::Kind::Int32:
			case BasicType::Kind::Int64:
				return true;
		}
	}
	return false;
}

static bool IsFloatType(const Type *type)
{                       
	if (auto basicType = GetBasicType(type); basicType != nullptr)
	{
		auto kind = basicType->GetKind();
		return (kind == BasicType::Kind::Float32 || kind == BasicType::Kind::Float64);
	}
	return false;
}

static bool IsComplexType(const Type *type)
{
	return IsBasicType(type, BasicType::Kind::Complex);
}

static bool IsRealType(const Type *type)
{
	return IsIntegerType(type) || IsFloatType(type);
}

static bool IsNumericType(const Type *type)
{
	return IsRealType(type) || IsComplexType(type);
}

static bool IsExtendedType(const Type *type)
{                       
	if (auto basicType = GetBasicType(type); basicType != nullptr)
	{
		auto kind = basicType->GetKind();
		return (kind == BasicType::Kind::Int64 || kind == BasicType::Kind::Float64);
	}
	return false;
}

static bool IsStringType(const Type *type)
{
	return IsBasicType(type, BasicType::Kind::String);
}

static bool IsSymbolType(const Type *type)
{
	return IsBasicType(type, BasicType::Kind::Symbol);
}

static bool IsCharacterType(const Type *type)
{
	return IsStringType(type) || IsSymbolType(type);
}

static bool IsCalendarType(const Type *type)
{
	if (auto basicType = GetBasicType(type); basicType != nullptr)
	{
		switch (basicType->GetKind())
		{
			case BasicType::Kind::Datetime:
			case BasicType::Kind::Date:
			case BasicType::Kind::Month:
			case BasicType::Kind::Time:
			case BasicType::Kind::Minute:
			case BasicType::Kind::Second:
				return true;
		}
	}
	return false;
}

static bool IsDatetimeType(const Type *type)
{
	return IsBasicType(type, BasicType::Kind::Datetime);
}

static bool IsDateType(const Type *type)
{
	return IsBasicType(type, BasicType::Kind::Date);
}

static bool IsMonthType(const Type *type)
{
	return IsBasicType(type, BasicType::Kind::Month);
}

static bool IsTimeType(const Type *type)
{
	return IsBasicType(type, BasicType::Kind::Time);
}

static bool IsMinuteType(const Type *type)
{
	return IsBasicType(type, BasicType::Kind::Minute);
}

static bool IsSecondType(const Type *type)
{
	return IsBasicType(type, BasicType::Kind::Second);
}

static bool IsAssignableType(const Type *type1, const Type *type2)
{
	if (*type1 == *type2)
	{
		return true;
	}

	if (IsNumericType(type1) && IsNumericType(type2))
	{
		auto basicType1 = GetBasicType(type1);
		auto basicType2 = GetBasicType(type2);

		auto kind1 = static_cast<std::underlying_type<BasicType::Kind>::type>(basicType1->GetKind());
		auto kind2 = static_cast<std::underlying_type<BasicType::Kind>::type>(basicType2->GetKind());

		return kind1 > kind2;
	}
	return false;
}

static bool IsOrderableTypes(const Type *type1, const Type *type2)
{
	if (IsRealType(type1) && IsRealType(type2))
	{
		return true;
	}
	if (*type1 == *type2 && (IsCalendarType(type1) || IsCharacterType(type1)))
	{
		return true;
	}
	return false;
}

static bool IsComparableTypes(const Type *type1, const Type *type2)
{
	if (IsRealType(type1) && IsRealType(type2))
	{
		return true;
	}
	if (*type1 == *type2 && (IsCalendarType(type1) || IsCharacterType(type1)))
	{
		return true;
	}
	return false;
}

static bool IsValueType(const Type *type)
{
	if (auto basicType = GetBasicType(type); basicType != nullptr)
	{
		return basicType->GetKind() != BasicType::Kind::Function;
	}
	return false;
}

static bool IsFunctionType(const Type *type)
{
	return IsBasicType(type, BasicType::Kind::Function);
}

static BasicType *WidestType(const BasicType *type1, const BasicType *type2)
{
	if (type1->GetKind() == BasicType::Kind::Complex ||
		type2->GetKind() == BasicType::Kind::Complex)
	{
		return new BasicType(BasicType::Kind::Complex);
	}
	else if (type1->GetKind() == BasicType::Kind::Float64 ||
		type2->GetKind() == BasicType::Kind::Float64)
	{
		return new BasicType(BasicType::Kind::Float64);
	}
	else if (type1->GetKind() == BasicType::Kind::Float32 ||
		type2->GetKind() == BasicType::Kind::Float32)
	{
		return new BasicType(BasicType::Kind::Float32);
	}
	else if (type1->GetKind() == BasicType::Kind::Int64 ||
		type2->GetKind() == BasicType::Kind::Int64)
	{
		return new BasicType(BasicType::Kind::Int64);
	}
	else if (type1->GetKind() == BasicType::Kind::Int32 ||
		type2->GetKind() == BasicType::Kind::Int32)
	{
		return new BasicType(BasicType::Kind::Int32);
	}
	else if (type1->GetKind() == BasicType::Kind::Int16 ||
		type2->GetKind() == BasicType::Kind::Int16)
	{
		return new BasicType(BasicType::Kind::Int16);
	}
	else if (type1->GetKind() == BasicType::Kind::Int8 ||
		type2->GetKind() == BasicType::Kind::Int8)
	{
		return new BasicType(BasicType::Kind::Int8);
	}
	Utils::Logger::LogError("Unknown widest type for " + type1->ToString() + " and " + type2->ToString());
}

static BasicType *WidestType(const Type *type1, const Type *type2)
{
	if (IsBasicType(type1) && IsBasicType(type2))
	{
		return WidestType(static_cast<const BasicType *>(type1), static_cast<const BasicType *>(type2));
	}
	Utils::Logger::LogError("Unknown widest type for " + type1->ToString() + " and " + type2->ToString());
}

}
