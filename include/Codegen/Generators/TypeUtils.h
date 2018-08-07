#pragma once

#include "HorseIR/Tree/Types/Type.h"
#include "HorseIR/Tree/Types/BasicType.h"

#include "Utils/Logger.h"

namespace Codegen {

static HorseIR::BasicType *WidestType(const HorseIR::BasicType *type1, const HorseIR::BasicType *type2)
{
	if (type1->GetKind() == HorseIR::BasicType::Kind::Float64 ||
		type2->GetKind() == HorseIR::BasicType::Kind::Float64)
	{
		return new HorseIR::BasicType(HorseIR::BasicType::Kind::Float64);
	}
	else if (type1->GetKind() == HorseIR::BasicType::Kind::Float32 ||
		type2->GetKind() == HorseIR::BasicType::Kind::Float32)
	{
		return new HorseIR::BasicType(HorseIR::BasicType::Kind::Float32);
	}
	else if (type1->GetKind() == HorseIR::BasicType::Kind::Int64 ||
		type2->GetKind() == HorseIR::BasicType::Kind::Int64)
	{
		return new HorseIR::BasicType(HorseIR::BasicType::Kind::Int64);
	}
	else if (type1->GetKind() == HorseIR::BasicType::Kind::Int32 ||
		type2->GetKind() == HorseIR::BasicType::Kind::Int32)
	{
		return new HorseIR::BasicType(HorseIR::BasicType::Kind::Int32);
	}
	else if (type1->GetKind() == HorseIR::BasicType::Kind::Int16 ||
		type2->GetKind() == HorseIR::BasicType::Kind::Int16)
	{
		return new HorseIR::BasicType(HorseIR::BasicType::Kind::Int16);
	}
	else if (type1->GetKind() == HorseIR::BasicType::Kind::Int8 ||
		type2->GetKind() == HorseIR::BasicType::Kind::Int8)
	{
		return new HorseIR::BasicType(HorseIR::BasicType::Kind::Int8);
	}
	Utils::Logger::LogError("Unknown widest type for " + type1->ToString() + " and " + type2->ToString());
}

static HorseIR::BasicType *WidestType(const HorseIR::Type *type1, const HorseIR::Type *type2)
{
	if (type1->GetKind() == HorseIR::Type::Kind::Basic &&
		type2->GetKind() == HorseIR::Type::Kind::Basic)
	{
		return WidestType(static_cast<const HorseIR::BasicType *>(type1), static_cast<const HorseIR::BasicType *>(type2));
	}
	Utils::Logger::LogError("Unknown widest type for " + type1->ToString() + " and " + type2->ToString());
}

}
