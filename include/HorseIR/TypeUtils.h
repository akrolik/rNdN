#pragma once

#include "HorseIR/Tree/Types/Type.h"
#include "HorseIR/Tree/Types/BasicType.h"

#include "Utils/Logger.h"

namespace HorseIR {

static bool IsBasicType(const Type *type, BasicType::Kind kind)
{
	if (type->GetKind() != Type::Kind::Basic)
	{
		return false;
	}

	auto basicType = static_cast<const BasicType *>(type);
	return (basicType->GetKind() == kind);
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

static bool IsBoolType(const Type *type)
{                       
	return IsBasicType(type, BasicType::Kind::Bool);
}

static bool IsRealType(const Type *type)
{                       
	if (type->GetKind() != Type::Kind::Basic)
	{
		return false;
	}

	auto basicType = static_cast<const BasicType *>(type);
	switch (basicType->GetKind())
	{
		case BasicType::Kind::Bool:
		case BasicType::Kind::Int8:
		case BasicType::Kind::Int16:
		case BasicType::Kind::Int32:
		case BasicType::Kind::Int64:
		case BasicType::Kind::Float32:
		case BasicType::Kind::Float64:
			return true;
	}
	return false;
}

static bool IsIntegerType(const Type *type)
{                       
	if (type->GetKind() != Type::Kind::Basic)
	{
		return false;
	}

	auto basicType = static_cast<const BasicType *>(type);
	switch (basicType->GetKind())
	{
		case BasicType::Kind::Bool:
		case BasicType::Kind::Int8:
		case BasicType::Kind::Int16:
		case BasicType::Kind::Int32:
		case BasicType::Kind::Int64:
			return true;
	}
	return false;
}

static bool IsFloatType(const Type *type)
{                       
	if (type->GetKind() != Type::Kind::Basic)
	{
		return false;
	}

	auto basicType = static_cast<const BasicType *>(type);
	auto kind = basicType->GetKind();

	return (kind == BasicType::Kind::Float32 || kind == BasicType::Kind::Float64);
}

static bool IsComplexType(const Type *type)
{
	return IsBasicType(type, BasicType::Kind::Complex);
}

static bool IsExtendedType(const Type *type)
{                       
	if (type->GetKind() != Type::Kind::Basic)
	{
		return false;
	}

	auto basicType = static_cast<const BasicType *>(type);
	auto kind = basicType->GetKind();

	return (kind == BasicType::Kind::Int64 || kind == BasicType::Kind::Float64);
}

static bool IsComparableTypes(const Type *type1, const Type *type2)
{
	if (IsRealType(type1) && IsRealType(type2))
	{
		return true;
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

static bool IsFunctionType(const Type *type)
{
	return IsBasicType(type, BasicType::Kind::Function);
}

static BasicType *WidestType(const BasicType *type1, const BasicType *type2)
{
	if (type1->GetKind() == BasicType::Kind::Float64 ||
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
