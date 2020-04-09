#pragma once

#include <vector>

#include "HorseIR/Tree/Tree.h"
#include "HorseIR/Utils/PrettyPrinter.h"

#include "Utils/Logger.h"

namespace HorseIR {

class TypeUtils {
public:

static std::string TypeString(const Type *type)
{
	return TypeString({type});
}

static std::string TypeString(const std::vector<Type *>& types)
{
	std::string s;
	bool first = true;
	for (const auto type : types)
	{
		if (!first)
		{
			s += ", ";
		}
		s += "'" + PrettyPrinter::PrettyString(type) + "'";
		first = false;
	}
	return s;
}

static bool IsEmptyType(const std::vector<Type *>& types)
{
	return (types.size() == 0);
}

static bool IsSingleType(const std::vector<Type *>& types)
{
	return (types.size() == 1);
}

static Type *GetSingleType(const std::vector<Type *>& types)
{
	if (types.size() == 1)
	{
		return types.at(0);
	}
	return nullptr;
}

static bool IsTypesAssignable(const std::vector<Type *>& types1, const std::vector<Type *>& types2)
{
	if (types1.size() != types2.size())
	{
		return false;
	}

	for (unsigned int i = 0; i < types1.size(); ++i)
	{
		auto type1 = types1.at(i);
		auto type2 = types2.at(i);
		if (type2 == nullptr)
		{
			continue;
		}

		if (IsType<WildcardType>(type1))
		{
			//TODO: Wildcard assign
		}

		if (IsType<ListType>(type1) && IsType<ListType>(type2))
		{
			auto listType1 = GetType<ListType>(type1);
			auto listType2 = GetType<ListType>(type2);

			if (const auto elementType = TypeUtils::GetReducedType(listType2->GetElementTypes()))
			{
				if (IsTypesEqual({elementType}, listType1->GetElementTypes()))
				{
					continue;
				}
				return false;
			}
		}

		if (!IsTypesEqual(type1, type2))
		{
			return false;
		}
	}
	return true;
}

static bool IsTypesEqual(const Type *type1, const Type *type2)
{
	return *type1 == *type2;
}

static bool IsTypesEqual(const std::vector<Type *>& types1, const std::vector<Type *>& types2, bool allowNull = false)
{
	if (types1.size() != types2.size())
	{
		return false;
	}
	for (unsigned int i = 0; i < types1.size(); ++i)
	{
		auto type1 = types1.at(i);
		auto type2 = types2.at(i);
		if (type1 == nullptr || type2 == nullptr)
		{
			if (allowNull)
			{
				continue;
			}
			return false;
		}

		if (!IsTypesEqual(type1, type2))
		{
			return false;
		}
	}
	return true;
}

static bool IsReducibleTypes(const std::vector<Type *>& types)
{
	Type *firstType = nullptr;
	for (const auto type : types)
	{
		if (firstType == nullptr)
		{
			firstType = type;
		}
		else if (*firstType != *type)
		{
			return false;
		}
	}
	return true;
}

static Type *GetReducedType(const std::vector<Type *>& types)
{
	if (IsReducibleTypes(types))
	{
		return types.at(0);
	}
	return nullptr;
}

static bool IsCastable(const Type *destination, const Type *source)
{
	if (source == nullptr)
	{
		return true;
	}

	if (*destination == *source)
	{
		return true;
	}

	if (!IsType<BasicType>(source) || !IsType<BasicType>(destination))
	{
		return false;
	}

	const auto basicDestination = GetType<BasicType>(destination);
	const auto basicSource = GetType<BasicType>(source);

	if (IsIntegerType(basicDestination))
	{
		if (IsIntegerType(basicSource))
		{
			// Allow if destination is wider than source

			return (GetBitSize(basicDestination) >= GetBitSize(basicSource));
		}
		else if (IsFloatType(basicSource))
		{
			// Truncate decimal part

			return (GetBitSize(basicDestination) >= GetBitSize(basicSource));
		}
		else if (IsBooleanType(basicSource))
		{
			// Always allowed (0-false, 1-true)

			return true;
		}
		return false;
	}
	else if (IsFloatType(basicDestination))
	{
		if (IsIntegerType(basicSource))
		{
			// Always allowed, but loss of precision may occur
			
			return true;
		}
		else if (IsFloatType(basicSource))
		{
			// Allow if destination is wider than source

			return (GetBitSize(basicDestination) >= GetBitSize(basicSource));
		}
		return false;
	}
	else if (IsBooleanType(basicDestination))
	{
		if (IsIntegerType(basicSource))
		{
			// Always allowed (0-false, 1-true)

			return true;
		}
		return false;
	}
	else if (IsCharacterType(basicDestination))
	{
		// Symbol and string may be cast to each other

		return IsCharacterType(basicSource);
	}

	return false;
}

static bool ForallElements(const ListType *listType, bool (*f)(const Type *))
{
	for (const auto elementType : listType->GetElementTypes())
	{
		if (!(*f)(elementType))
		{
			return false;
		}
	}
	return true;
}

static bool ForanyElement(const ListType *listType, bool (*f)(const Type *))
{
	for (const auto elementType : listType->GetElementTypes())
	{
		if ((*f)(elementType))
		{
			return true;
		}
	}
	return false;
}

template<class T>
static T *GetType(Type *type)
{
	if (type->m_kind == T::TypeKind)
	{
		return static_cast<T *>(type);
	}
	return nullptr;
}

template<class T>
static const T *GetType(const Type *type)
{
	if (type->m_kind == T::TypeKind)
	{
		return static_cast<const T *>(type);
	}
	return nullptr;
}

template<class T>
static bool IsType(const Type *type)
{
	return (type->m_kind == T::TypeKind);
}

static bool IsBasicType(const Type *type, BasicType::BasicKind kind)
{
	if (const auto basicType = GetType<BasicType>(type))
	{
		return (basicType->GetBasicKind() == kind);
	}
	return false;
}

static int GetBitSize(const BasicType *type)
{                       
	switch (type->GetBasicKind())
	{
		case BasicType::BasicKind::Boolean:
			return 1;
		case BasicType::BasicKind::Char:
			return 8;
		case BasicType::BasicKind::Int8:
			return 8;
		case BasicType::BasicKind::Int16:
			return 16;
		case BasicType::BasicKind::Int32:
			return 32;
		case BasicType::BasicKind::Int64:
			return 64;
		case BasicType::BasicKind::Float32:
			return 32;
		case BasicType::BasicKind::Float64:
			return 64;
	}
	return 0;
}

static bool IsBooleanType(const Type *type)
{                       
	return IsBasicType(type, BasicType::BasicKind::Boolean);
}

static bool IsIntegerType(const Type *type)
{                       
	if (const auto basicType = GetType<BasicType>(type))
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
	if (const auto basicType = GetType<BasicType>(type))
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
	if (const auto basicType = GetType<BasicType>(type))
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
	if (const auto basicType = GetType<BasicType>(type))
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

		auto kind1 = static_cast<std::underlying_type<BasicType::BasicKind>::type>(basicType1->GetBasicKind());
		auto kind2 = static_cast<std::underlying_type<BasicType::BasicKind>::type>(basicType2->GetBasicKind());

		return kind1 > kind2;
	}
	return false;
}

static bool IsOrderableType(const Type *type)
{
	return (IsRealType(type) || IsCharacterType(type) || IsCalendarType(type));
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
	if (type1->GetBasicKind() == type2->GetBasicKind())
	{
		return new BasicType(type1->GetBasicKind());
	}
	else if (type1->GetBasicKind() == BasicType::BasicKind::Complex ||
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
	Utils::Logger::LogError("Unknown widest type for " + PrettyPrinter::PrettyString(type1) + " and " + PrettyPrinter::PrettyString(type2));
}

static BasicType *WidestType(const Type *type1, const Type *type2)
{
	if (IsType<BasicType>(type1) && IsType<BasicType>(type2))
	{
		return WidestType(GetType<BasicType>(type1), GetType<BasicType>(type2));
	}
	Utils::Logger::LogError("Unknown widest type for " + PrettyPrinter::PrettyString(type1) + " and " + PrettyPrinter::PrettyString(type2));
}

};

}
