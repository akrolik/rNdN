#pragma once

#include "HorseIR/Analysis/PrettyPrinter.h"
#include "HorseIR/Tree/Types/Type.h"
#include "HorseIR/Tree/Types/BasicType.h"

#include "Utils/Logger.h"

namespace HorseIR {

template<class T>
static T *GetType(Type *type)
{
	if (type->GetKind() == T::TypeKind)
	{
		return static_cast<T *>(type);
	}
	return nullptr;
}

template<class T>
static const T *GetType(const Type *type)
{
	if (type->GetKind() == T::TypeKind)
	{
		return static_cast<const T *>(type);
	}
	return nullptr;
}

template<class T>
static bool IsType(const Type *type)
{
	return (type->GetKind() == T::TypeKind);
}

static bool IsBasicType(const Type *type, BasicType::BasicKind kind)
{
	if (auto basicType = GetType<BasicType>(type); basicType != nullptr)
	{
		return (basicType->GetBasicKind() == kind);
	}
	return false;
}

static bool IsBooleanType(const Type *type)
{                       
	return IsBasicType(type, BasicType::BasicKind::Boolean);
}

static bool IsIntegerType(const Type *type)
{                       
	if (auto basicType = GetType<BasicType>(type); basicType != nullptr)
	{
		switch (basicType->GetBasicKind())
		{
			case BasicType::BasicKind::Boolean:
			case BasicType::BasicKind::Char:
			case BasicType::BasicKind::Int8:
			case BasicType::BasicKind::Int16:
			case BasicType::BasicKind::Int32:
			case BasicType::BasicKind::Int64:
				return true;
		}
	}
	return false;
}

static bool IsFloatType(const Type *type)
{                       
	if (auto basicType = GetType<BasicType>(type); basicType != nullptr)
	{
		auto kind = basicType->GetBasicKind();
		return (kind == BasicType::BasicKind::Float32 || kind == BasicType::BasicKind::Float64);
	}
	return false;
}

static bool IsComplexType(const Type *type)
{
	return IsBasicType(type, BasicType::BasicKind::Complex);
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
	if (auto basicType = GetType<BasicType>(type); basicType != nullptr)
	{
		auto kind = basicType->GetBasicKind();
		return (kind == BasicType::BasicKind::Int64 || kind == BasicType::BasicKind::Float64);
	}
	return false;
}

static bool IsStringType(const Type *type)
{
	return IsBasicType(type, BasicType::BasicKind::String);
}

static bool IsSymbolType(const Type *type)
{
	return IsBasicType(type, BasicType::BasicKind::Symbol);
}

static bool IsCharacterType(const Type *type)
{
	return IsStringType(type) || IsSymbolType(type);
}

static bool IsCalendarType(const Type *type)
{
	if (auto basicType = GetType<BasicType>(type); basicType != nullptr)
	{
		switch (basicType->GetBasicKind())
		{
			case BasicType::BasicKind::Datetime:
			case BasicType::BasicKind::Date:
			case BasicType::BasicKind::Month:
			case BasicType::BasicKind::Time:
			case BasicType::BasicKind::Minute:
			case BasicType::BasicKind::Second:
				return true;
		}
	}
	return false;
}

static bool IsDatetimeType(const Type *type)
{
	return IsBasicType(type, BasicType::BasicKind::Datetime);
}

static bool IsDateType(const Type *type)
{
	return IsBasicType(type, BasicType::BasicKind::Date);
}

static bool IsMonthType(const Type *type)
{
	return IsBasicType(type, BasicType::BasicKind::Month);
}

static bool IsTimeType(const Type *type)
{
	return IsBasicType(type, BasicType::BasicKind::Time);
}

static bool IsMinuteType(const Type *type)
{
	return IsBasicType(type, BasicType::BasicKind::Minute);
}

static bool IsSecondType(const Type *type)
{
	return IsBasicType(type, BasicType::BasicKind::Second);
}

static bool IsAssignableType(const Type *type1, const Type *type2)
{
	if (*type1 == *type2)
	{
		return true;
	}

	if (IsNumericType(type1) && IsNumericType(type2))
	{
		auto basicType1 = GetType<BasicType>(type1);
		auto basicType2 = GetType<BasicType>(type2);

		auto kind1 = static_cast<std::underlying_type<BasicType::BasicKind>::type>(basicType1->GetKind());
		auto kind2 = static_cast<std::underlying_type<BasicType::BasicKind>::type>(basicType2->GetKind());

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

static BasicType *WidestType(const BasicType *type1, const BasicType *type2)
{
	if (type1->GetBasicKind() == BasicType::BasicKind::Complex ||
		type2->GetBasicKind() == BasicType::BasicKind::Complex)
	{
		return new BasicType(BasicType::BasicKind::Complex);
	}
	else if (type1->GetBasicKind() == BasicType::BasicKind::Float64 ||
		type2->GetBasicKind() == BasicType::BasicKind::Float64)
	{
		return new BasicType(BasicType::BasicKind::Float64);
	}
	else if (type1->GetBasicKind() == BasicType::BasicKind::Float32 ||
		type2->GetBasicKind() == BasicType::BasicKind::Float32)
	{
		return new BasicType(BasicType::BasicKind::Float32);
	}
	else if (type1->GetBasicKind() == BasicType::BasicKind::Int64 ||
		type2->GetBasicKind() == BasicType::BasicKind::Int64)
	{
		return new BasicType(BasicType::BasicKind::Int64);
	}
	else if (type1->GetBasicKind() == BasicType::BasicKind::Int32 ||
		type2->GetBasicKind() == BasicType::BasicKind::Int32)
	{
		return new BasicType(BasicType::BasicKind::Int32);
	}
	else if (type1->GetBasicKind() == BasicType::BasicKind::Int16 ||
		type2->GetBasicKind() == BasicType::BasicKind::Int16)
	{
		return new BasicType(BasicType::BasicKind::Int16);
	}
	else if (type1->GetBasicKind() == BasicType::BasicKind::Int8 ||
		type2->GetBasicKind() == BasicType::BasicKind::Int8)
	{
		return new BasicType(BasicType::BasicKind::Int8);
	}
	else if (type1->GetBasicKind() == BasicType::BasicKind::Char ||
		type2->GetBasicKind() == BasicType::BasicKind::Char)
	{
		return new BasicType(BasicType::BasicKind::Char);
	}
	else if (type1->GetBasicKind() == BasicType::BasicKind::Boolean ||
		type2->GetBasicKind() == BasicType::BasicKind::Boolean)
	{
		return new BasicType(BasicType::BasicKind::Boolean);
	}
	Utils::Logger::LogError("Unknown widest type for " + PrettyPrinter::PrintString(type1) + " and " + PrettyPrinter::PrintString(type2));
}

static BasicType *WidestType(const Type *type1, const Type *type2)
{
	if (IsType<BasicType>(type1) && IsType<BasicType>(type2))
	{
		return WidestType(GetType<BasicType>(type1), GetType<BasicType>(type2));
	}
	Utils::Logger::LogError("Unknown widest type for " + PrettyPrinter::PrintString(type1) + " and " + PrettyPrinter::PrintString(type2));
}

}
